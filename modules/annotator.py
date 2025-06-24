# Manual Annotation Tool for Object Detection
# This tool allows users to annotate images or video frames with bounding boxes or oriented bounding boxes (OBBs).
# It supports undo/redo functionality, class selection, and saving annotations in JSON format.
# The tool is built using PySide6 for the GUI and OpenCV for image processing.
# It can handle both static images and video files, allowing users to navigate through frames and annotate them.
# The annotations can be saved to a specified output path, and users can also extract observation windows.

import os
import json
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QSpinBox,
                               QSlider, QComboBox, QFrame, QSplitter, QFileDialog)
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QFont, QPolygonF, QImage
import cv2
import sys
import copy

class AnnotationCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.tracks_data = {}
        self.current_frame = 0
        self.selected_track = None
        self.current_class = 10
        self.annotation_mode = "rect"
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.dragging = False
        self.drag_corner = None
        self.rotating = False
        self.rotation_start_angle = 0
        self.hover_track = None
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.history = []
        self.history_index = -1
        self.max_history = 50
        self.resizing = False
        self.resize_handle = None

    def save_state(self):
        state = json.dumps(self.tracks_data, sort_keys=True)
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        if not self.history or self.history[-1] != state:
            self.history.append(state)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            else:
                self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.tracks_data = json.loads(self.history[self.history_index])
            self.selected_track = None
            self.update()
            return True
        return False

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.tracks_data = json.loads(self.history[self.history_index])
            self.selected_track = None
            self.update()
            return True
        return False

    def set_image(self, image_path):
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img_rgb.shape
                bpl = 3 * w
                q_image = QImage(img_rgb.data, w, h, bpl, QImage.Format_RGB888)
                self.image = QPixmap.fromImage(q_image)
                self.fit_to_window()
                self.update()

    def set_frame(self, frame_array):
        if frame_array is not None:
            h, w, c = frame_array.shape
            bpl = 3 * w
            q_image = QImage(frame_array.data, w, h, bpl, QImage.Format_RGB888)
            self.image = QPixmap.fromImage(q_image)
            self.update()

    def fit_to_window(self):
        if not self.image:
            return
        widget_size = self.size()
        image_size = self.image.size()
        sx = widget_size.width() / image_size.width()
        sy = widget_size.height() / image_size.height()
        self.scale_factor = min(sx, sy, 1.0)
        scaled = image_size * self.scale_factor
        self.offset = QPoint((widget_size.width() - scaled.width()) // 2,
                             (widget_size.height() - scaled.height()) // 2)

    def screen_to_image(self, sp):
        if not self.image:
            return QPointF(0,0)
        return QPointF((sp.x()-self.offset.x())/self.scale_factor,
                       (sp.y()-self.offset.y())/self.scale_factor)

    def image_to_screen(self, ip):
        if not self.image:
            return QPoint(0,0)
        return QPoint(int(ip.x()*self.scale_factor + self.offset.x()),
                      int(ip.y()*self.scale_factor + self.offset.y()))

    def get_frame_detections(self):
        fd = {}
        if 'tracks' in self.tracks_data:
            for tid, td in self.tracks_data['tracks'].items():
                if td.get('first_frame',0) <= self.current_frame <= td.get('last_frame',0):
                    for det in td.get('detections',[]):
                        if det['frame']==self.current_frame:
                            fd[tid]=det
                            break
        return fd

    def _calculate_obb_corners(self, center, size, rot):
        cx, cy = center
        w, h = size
        cr, sr = math.cos(rot), math.sin(rot)
        base = [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]]
        res=[]
        for x,y in base:
            res.append([x*cr - y*sr + cx, x*sr + y*cr + cy])
        return res

    def get_obb_from_corners(self, corners):
        arr = np.array(corners)
        center = arr.mean(axis=0)
        e1 = arr[1]-arr[0]
        e2 = arr[3]-arr[0]
        w=np.linalg.norm(e1); h=np.linalg.norm(e2)
        rot = math.atan2(e1[1], e1[0])
        return center.tolist(),[w,h],rot

    def point_in_polygon(self, p, poly):
        x,y = p.x(),p.y(); inside=False
        n=len(poly); p1x,p1y=poly[0]
        for i in range(1,n+1):
            p2x,p2y=poly[i%n]
            if y>min(p1y,p2y):
                if y<=max(p1y,p2y):
                    if x<=max(p1x,p2x):
                        if p1y!=p2y:
                            xin=(y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x==p2x or x<=xin:
                            inside=not inside
            p1x,p1y=p2x,p2y
        return inside

    def get_closest_corner(self, p, corners):
        dists=[math.hypot(p.x()-c[0],p.y()-c[1]) for c in corners]
        return dists.index(min(dists))

    def find_box_at_point(self, p):
        fd=self.get_frame_detections()
        for tid,det in fd.items():
            if 'obb' in det:
                poly=det['obb']['corners']
                if self.point_in_polygon(p,poly):
                    return tid,det
            elif 'bbox' in det:
                x1,y1,x2,y2=det['bbox']
                if x1<=p.x()<=x2 and y1<=p.y()<=y2:
                    return tid,det
        return None,None

    def get_resize_handle(self, p, bbox):
        x1, y1, x2, y2 = bbox
        handle_size = 10 / self.scale_factor
        handles = {
            'tl': QPointF(x1, y1), 'tm': QPointF((x1 + x2) / 2, y1), 'tr': QPointF(x2, y1),
            'ml': QPointF(x1, (y1 + y2) / 2), 'mr': QPointF(x2, (y1 + y2) / 2),
            'bl': QPointF(x1, y2), 'bm': QPointF((x1 + x2) / 2, y2), 'br': QPointF(x2, y2)
        }
        for name, pos in handles.items():
            if math.hypot(p.x() - pos.x(), p.y() - pos.y()) < handle_size:
                return name
        return None

    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            ip=self.screen_to_image(e.position())
            tid,det=self.find_box_at_point(ip)
            if tid:
                self.selected_track=tid
                if 'obb' in det:
                    corners=det['obb']['corners']
                    ci=self.get_closest_corner(ip,corners)
                    cd=math.hypot(ip.x()-corners[ci][0],ip.y()-corners[ci][1])
                    if cd<15:
                        self.dragging=True; self.drag_corner=ci; self.update(); return
                    cen=det['obb']['center']
                    cdist=math.hypot(ip.x()-cen[0],ip.y()-cen[1])
                    if cdist<30:
                        self.rotating=True
                        self.rotation_start_angle=math.atan2(ip.y()-cen[1],ip.x()-cen[0])
                        self.update(); return
                elif 'bbox' in det:
                    handle = self.get_resize_handle(ip, det['bbox'])
                    if handle:
                        self.resizing = True
                        self.resize_handle = handle
                        self.update()
                        return
                self.update()
            else:
                self.drawing=True
                self.start_point=ip
                self.current_point=ip
                self.selected_track=None

    def mouseMoveEvent(self,e):
        ip=self.screen_to_image(e.position())
        if self.drawing:
            self.current_point=ip; self.update()
        elif self.resizing and self.selected_track:
            fd=self.get_frame_detections()
            if self.selected_track in fd:
                det=fd[self.selected_track]
                if 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    if 't' in self.resize_handle: y1 = ip.y()
                    if 'b' in self.resize_handle: y2 = ip.y()
                    if 'l' in self.resize_handle: x1 = ip.x()
                    if 'r' in self.resize_handle: x2 = ip.x()
                    det['bbox'] = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
                    self.update_track_detection(self.selected_track,self.current_frame,det)
                    self.update()
        elif self.rotating and self.selected_track:
            fd=self.get_frame_detections()
            if self.selected_track in fd and 'obb' in fd[self.selected_track]:
                det=fd[self.selected_track]
                cen=det['obb']['center']
                ca=math.atan2(ip.y()-cen[1],ip.x()-cen[0])
                diff=ca-self.rotation_start_angle
                newr=det['obb']['rotation']+diff
                corners=self._calculate_obb_corners(cen,det['obb']['size'],newr)
                det['obb']['rotation']=newr
                det['obb']['corners']=corners
                self.update_track_detection(self.selected_track,self.current_frame,det)
                self.rotation_start_angle=ca; self.update()
        elif self.dragging and self.selected_track is not None:
            fd=self.get_frame_detections()
            if self.selected_track in fd and 'obb' in fd[self.selected_track]:
                det=fd[self.selected_track]
                corners=det['obb']['corners']
                corners[self.drag_corner]=[ip.x(),ip.y()]
                cen,sz,rot=self.get_obb_from_corners(corners)
                det['obb']['center']=cen; det['obb']['size']=sz; det['obb']['rotation']=rot; det['obb']['corners']=corners
                self.update_track_detection(self.selected_track,self.current_frame,det)
                self.update()
        else:
            tid,_=self.find_box_at_point(ip)
            if tid!=self.hover_track:
                self.hover_track=tid; self.update()

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.LeftButton:
            if self.drawing and self.start_point and self.current_point:
                self.save_state()
                self.create_new_detection()
                self.drawing=False
                self.start_point=None
                self.current_point=None
            elif self.dragging or self.rotating:
                self.save_state()
                self.dragging=False
                self.drag_corner=None
                self.rotating=False
            elif self.resizing:
                self.save_state()
                self.resizing = False
                self.resize_handle = None

    def create_new_detection(self):
        if not (self.start_point and self.current_point): return
        if self.annotation_mode=="rect":
            x1=min(self.start_point.x(),self.current_point.x())
            x2=max(self.start_point.x(),self.current_point.x())
            y1=min(self.start_point.y(),self.current_point.y())
            y2=max(self.start_point.y(),self.current_point.y())
            if x2-x1>10 and y2-y1>10:
                det={"frame":self.current_frame,"bbox":[x1,y1,x2,y2],"confidence":1.0,"class":self.current_class}
                self.add_new_track(det)
        else:
            cx=(self.start_point.x()+self.current_point.x())/2
            cy=(self.start_point.y()+self.current_point.y())/2
            w=abs(self.current_point.x()-self.start_point.x())
            h=abs(self.current_point.y()-self.start_point.y())
            if w>10 and h>10:
                corners=self._calculate_obb_corners([cx,cy],[w,h],0)
                det={"frame":self.current_frame,"obb":{"center":[cx,cy],"size":[w,h],"rotation":0,"corners":corners},"confidence":1.0,"class":self.current_class}
                self.add_new_track(det)
        self.update()

    def add_new_track(self,d):
        if 'tracks' not in self.tracks_data:
            self.tracks_data['tracks']={}
        tids=[int(t) for t in self.tracks_data['tracks']] if self.tracks_data['tracks'] else []
        nid=max(tids,default=0)+1
        self.tracks_data['tracks'][str(nid)]={"track_id":nid,"class":d['class'],"first_frame":d['frame'],"last_frame":d['frame'],"detections":[d]}

    def update_track_detection(self,tid,fn,ud):
        if 'tracks' in self.tracks_data and tid in self.tracks_data['tracks']:
            for i,det in enumerate(self.tracks_data['tracks'][tid]['detections']):
                if det['frame']==fn:
                    self.tracks_data['tracks'][tid]['detections'][i]=ud
                    break

    def delete_selected_box(self):
        if self.selected_track and 'tracks' in self.tracks_data and self.selected_track in self.tracks_data['tracks']:
            self.save_state()
            td=self.tracks_data['tracks'][self.selected_track]
            td['detections']=[d for d in td['detections'] if d['frame']!=self.current_frame]
            if not td['detections']:
                del self.tracks_data['tracks'][self.selected_track]
            else:
                frs=[d['frame'] for d in td['detections']]
                td['first_frame']=min(frs); td['last_frame']=max(frs)
            self.selected_track=None; self.update()

    def keyPressEvent(self,e):
        if e.key()==Qt.Key_Delete:
            self.delete_selected_box()
        elif e.key()==Qt.Key_Escape:
            self.selected_track=None; self.drawing=False; self.dragging=False; self.rotating=False; self.resizing=False; self.update()
        elif e.key()==Qt.Key_Z and e.modifiers()&Qt.ControlModifier:
            if e.modifiers()&Qt.ShiftModifier: self.redo()
            else: self.undo()

    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if self.image:
            sz=self.image.size()*self.scale_factor
            tr=QRectF(self.offset.x(),self.offset.y(),sz.width(),sz.height())
            p.drawPixmap(tr,self.image,self.image.rect())
        self.draw_annotations(p)
        if self.drawing and self.start_point and self.current_point:
            self.draw_temp_box(p)

    def draw_annotations(self,p):
        fd=self.get_frame_detections()
        f=QFont(); f.setPointSize(10); p.setFont(f)
        for tid,det in fd.items():
            sel=tid==self.selected_track; hov=tid==self.hover_track
            if sel: col=QColor(0,255,255); w=3
            elif hov: col=QColor(255,255,0); w=2
            else: col=QColor(0,255,0); w=2
            p.setPen(QPen(col,w))
            if 'obb' in det:
                crn=det['obb']['corners']
                sc=[self.image_to_screen(QPointF(x,y)) for x,y in crn]
                poly=QPolygonF([QPointF(pt.x(),pt.y()) for pt in sc])
                p.drawPolygon(poly)
                cs=self.image_to_screen(QPointF(*det['obb']['center']))
                lbl=f"ID:{tid} C:{det.get('class','N/A')}"
                p.drawText(cs.x(),cs.y(),lbl)
                if sel:
                    p.setPen(QPen(QColor(255,0,0),2)); p.setBrush(QBrush(QColor(255,0,0)))
                    for pt in sc: p.drawEllipse(pt,5,5)
                    p.setPen(QPen(QColor(255,255,0),3)); p.setBrush(QBrush(QColor(255,255,0,100)))
                    p.drawEllipse(cs,8,8)
            elif 'bbox' in det:
                x1,y1,x2,y2=det['bbox']
                tl=self.image_to_screen(QPointF(x1,y1))
                br=self.image_to_screen(QPointF(x2,y2))
                r=QRectF(tl, br)
                p.drawRect(r)
                lbl=f"ID:{tid} C:{det.get('class','N/A')}"
                p.drawText(tl.x(),tl.y()-5,lbl)
                if sel:
                    p.setPen(QPen(QColor(255,0,0),2))
                    p.setBrush(QBrush(QColor(255,0,0)))
                    handle_size = 4
                    handles = [
                        tl, QPoint((tl.x() + br.x()) // 2, tl.y()), QPoint(br.x(), tl.y()),
                        QPoint(tl.x(), (tl.y() + br.y()) // 2), QPoint(br.x(), (tl.y() + br.y()) // 2),
                        QPoint(tl.x(), br.y()), QPoint((tl.x() + br.x()) // 2, br.y()), br
                    ]
                    for handle in handles:
                        p.drawEllipse(handle, handle_size, handle_size)

    def draw_temp_box(self,p):
        p.setPen(QPen(QColor(255,0,0),2,Qt.DashLine))
        ss=self.image_to_screen(self.start_point)
        cs=self.image_to_screen(self.current_point)
        if self.annotation_mode=="rect":
            r=QRectF(ss, cs); p.drawRect(r.normalized())
        else:
            cx=(self.start_point.x()+self.current_point.x())/2
            cy=(self.start_point.y()+self.current_point.y())/2
            w=abs(self.current_point.x()-self.start_point.x())
            h=abs(self.current_point.y()-self.start_point.y())
            crn=self._calculate_obb_corners([cx,cy],[w,h],0)
            sc=[self.image_to_screen(QPointF(x,y)) for x,y in crn]
            poly=QPolygonF([QPointF(pt.x(),pt.y()) for pt in sc])
            p.drawPolygon(poly)

class ManualAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Annotation Tool")
        self.setGeometry(100,100,1200,800)
        self.source_path=None
        self.output_path=None
        self.cap=None
        self.is_video=False
        self.total_frames=1
        self.setup_ui()

    def setup_ui(self):
        cw=QWidget(); self.setCentralWidget(cw)
        ml=QHBoxLayout(cw)
        splitter=QSplitter(Qt.Horizontal)
        ml.addWidget(splitter)
        self.canvas=AnnotationCanvas()
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.create_control_panel())
        splitter.setStretchFactor(0,3)
        splitter.setStretchFactor(1,1)

    def create_control_panel(self):
        p=QFrame(); p.setFrameStyle(QFrame.StyledPanel); p.setMaximumWidth(250)
        l=QVBoxLayout(p)
        l.addWidget(QLabel("Frame Navigation"))
        self.frame_slider=QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        l.addWidget(self.frame_slider)
        self.frame_label=QLabel("Frame: 0/0"); l.addWidget(self.frame_label)
        nav=QHBoxLayout()
        self.prev_btn=QPushButton("Previous"); self.prev_btn.clicked.connect(self.previous_frame)
        self.next_btn=QPushButton("Next"); self.next_btn.clicked.connect(self.next_frame)
        nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn)
        l.addLayout(nav)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        l.addWidget(line)

        l.addWidget(QLabel("Annotation Mode"))
        self.mode_combo=QComboBox(); self.mode_combo.addItems(["rect","obb"])
        self.mode_combo.currentTextChanged.connect(lambda m: setattr(self.canvas,"annotation_mode",m))
        l.addWidget(self.mode_combo)
        l.addWidget(QLabel("Class"))
        self.class_spinbox=QSpinBox(); self.class_spinbox.setRange(0,999)
        self.class_spinbox.setValue(10)
        self.class_spinbox.valueChanged.connect(self.on_class_changed)
        l.addWidget(self.class_spinbox)
        l.addWidget(QLabel("Track ID"))
        self.track_spinbox=QSpinBox(); self.track_spinbox.setRange(0,9999)
        self.track_spinbox.valueChanged.connect(self.on_trackid_changed)
        l.addWidget(self.track_spinbox)
        el=QHBoxLayout()
        self.undo_btn=QPushButton("Undo"); self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn=QPushButton("Redo"); self.redo_btn.clicked.connect(self.canvas.redo)
        el.addWidget(self.undo_btn); el.addWidget(self.redo_btn)
        l.addLayout(el)
        self.delete_btn=QPushButton("Delete Selected"); self.delete_btn.clicked.connect(self.delete_selected)
        l.addWidget(self.delete_btn)
        self.save_btn=QPushButton("Save"); self.save_btn.clicked.connect(self.save_annotations)
        l.addWidget(self.save_btn)
        self.info_label=QLabel("Boxes: 0"); l.addWidget(self.info_label)
        
        line2 = QFrame(); line2.setFrameShape(QFrame.HLine); line2.setFrameShadow(QFrame.Sunken)
        l.addWidget(line2)
        
        l.addWidget(QLabel("Extract Observation Window"))
        self.start_frame_spinbox = QSpinBox()
        self.end_frame_spinbox = QSpinBox()
        l.addWidget(QLabel("Start Frame:"))
        l.addWidget(self.start_frame_spinbox)
        l.addWidget(QLabel("End Frame:"))
        l.addWidget(self.end_frame_spinbox)
        self.extract_btn = QPushButton("Extract Clip")
        self.extract_btn.clicked.connect(self.extract_observation)
        l.addWidget(self.extract_btn)

        l.addStretch()
        return p

    def load_media(self, source_path, output_path):
        self.source_path=source_path
        self.output_path=output_path
        jp=os.path.splitext(source_path)[0]+'.json'
        if os.path.exists(jp):
            with open(jp,'r') as f:
                self.canvas.tracks_data=json.load(f)
                self.canvas.save_state()
        if source_path.lower().endswith(('.mp4','.avi','.mov','.mkv')):
            self.cap=cv2.VideoCapture(source_path)
            self.is_video=True
            self.total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(self.total_frames-1)
            self.start_frame_spinbox.setRange(0, self.total_frames-1)
            self.end_frame_spinbox.setRange(0, self.total_frames-1)
            self.end_frame_spinbox.setValue(self.total_frames-1)
            self.extract_btn.setEnabled(True)
            self.load_frame(0)
        else:
            self.is_video=False
            self.total_frames=1
            self.canvas.set_image(source_path)
            self.canvas.current_frame=0
            self.start_frame_spinbox.setEnabled(False)
            self.end_frame_spinbox.setEnabled(False)
            self.extract_btn.setEnabled(False)
            self.canvas.update()
            self.update_info()

    def load_frame(self, fn):
        if self.is_video and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,fn)
            ret,frame=self.cap.read()
            if ret:
                fr_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                self.canvas.set_frame(fr_rgb)
                self.canvas.current_frame=fn
                self.update_info()

    def on_frame_changed(self,v):
        if self.is_video:
            self.load_frame(v)

    def previous_frame(self):
        if self.canvas.current_frame>0:
            self.frame_slider.setValue(self.canvas.current_frame-1)

    def next_frame(self):
        if self.canvas.current_frame<self.total_frames-1:
            self.frame_slider.setValue(self.canvas.current_frame+1)

    def on_class_changed(self,val):
        self.canvas.current_class=val
        if self.canvas.selected_track:
            tid=self.canvas.selected_track
            fd=self.canvas.get_frame_detections()
            det=fd.get(tid)
            if det:
                det['class']=val
                self.canvas.update_track_detection(tid,self.canvas.current_frame,det)
                self.canvas.update()

    def on_trackid_changed(self,val):
        old=self.canvas.selected_track
        if old and 'tracks' in self.canvas.tracks_data:
            td=self.canvas.tracks_data['tracks']
            if str(val) not in td:
                data=td.pop(old)
                data['track_id']=val
                td[str(val)]=data
                self.canvas.selected_track=str(val)
                self.canvas.update()
        self.update_info()

    def delete_selected(self):
        self.canvas.delete_selected_box()
        self.update_info()

    def save_annotations(self):
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
            with open(self.output_path,'w') as f:
                json.dump(self.canvas.tracks_data,f,indent=2)
            print(f"Saved annotations to {self.output_path}")
            
    def extract_observation(self):
        if not self.is_video or not self.cap:
            return

        start_frame = self.start_frame_spinbox.value()
        end_frame = self.end_frame_spinbox.value()

        if start_frame >= end_frame:
            print("Error: Start frame must be less than end frame.")
            return

        output_video_path, _ = QFileDialog.getSaveFileName(self, "Save Observation Video", "", "MP4 Files (*.mp4)")

        if not output_video_path:
            return

        output_json_path = os.path.splitext(output_video_path)[0] + '.json'

        new_tracks_data = {'tracks': {}}
        if 'tracks' in self.canvas.tracks_data:
            original_tracks = copy.deepcopy(self.canvas.tracks_data['tracks'])
            for track_id, track_data in original_tracks.items():
                new_detections = []
                for det in track_data.get('detections', []):
                    if start_frame <= det['frame'] <= end_frame:
                        new_det = det.copy()
                        new_det['frame'] -= start_frame
                        new_detections.append(new_det)
                
                if new_detections:
                    new_track = track_data.copy()
                    new_track['detections'] = new_detections
                    new_track['first_frame'] = min(d['frame'] for d in new_detections)
                    new_track['last_frame'] = max(d['frame'] for d in new_detections)
                    new_tracks_data['tracks'][track_id] = new_track

        with open(output_json_path, 'w') as f:
            json.dump(new_tracks_data, f, indent=2)
        print(f"Saved filtered annotations to {output_json_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_num in range(start_frame, end_frame + 1):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                out.write(frame)
        
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.canvas.current_frame)
        print(f"Saved extracted video to {output_video_path}")

    def update_info(self):
        fd=self.canvas.get_frame_detections()
        bc=len(fd)
        cf=self.canvas.current_frame+1 if self.total_frames > 0 else 0
        self.frame_label.setText(f"Frame: {cf}/{self.total_frames}")
        if self.canvas.selected_track and self.canvas.selected_track in fd:
            self.info_label.setText(f"Boxes: {bc} | Selected: ID {self.canvas.selected_track}")
            det=fd.get(self.canvas.selected_track)
            if det:
                self.class_spinbox.setValue(det.get('class',self.canvas.current_class))
            self.track_spinbox.setValue(int(self.canvas.selected_track))
            self.class_spinbox.setEnabled(True)
            self.track_spinbox.setEnabled(True)
        else:
            self.info_label.setText(f"Boxes: {bc}")
            self.class_spinbox.setEnabled(True)
            self.track_spinbox.setEnabled(False)

def manual_annotate(source_path, output_path, load_path=None):
    app=QApplication(sys.argv)
    win=ManualAnnotatorApp()
    if load_path and os.path.exists(load_path):
        with open(load_path,'r') as f:
            win.canvas.tracks_data=json.load(f)
            win.canvas.save_state()
    win.load_media(source_path,output_path)
    win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python annotator.py <source_path> <output_path> [load_path]")
        sys.exit(1)
    manual_annotate(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)