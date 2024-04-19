import os
import numpy as np
import cv2
import fitz
from PIL import Image
import yolov9
import bbox_visualizer2 as bbv

import torch


class CdrawingAnalyzer:

    def __init__(self, device='cpu'):
        self.device = device 

        ## legend detector model
        legend_detector_weights = './weights/legend_detection_best.pt'
        self.legend_model = yolov9.load(legend_detector_weights, device=device)

        ## hatches detector model
        hatch_detector_weights = './weights/hatch_detection_best.pt'
        self.hatch_model = yolov9.load(hatch_detector_weights, device=device)


    def calibrate_scale(self, drawing_pdf):
        self.drawing_filename = drawing_pdf

        ## load the pdf
        self.doc = fitz.open(self.drawing_filename)
        self.page = self.doc[0]

        ## text layer of pdf
        self.text = self.page.get_text()

        ## type of drawing
        if "RCP" in self.text or "Ceiling" in self.text:
            type_d = 'RCP'
        
        elif "SIGNAGE" in self.text:
            type_d = 'Signage'
        else:
            type_d = 'Other'


        sr_prefix = 'Scale 1 : '

        sr_instances = []

        for item in self.text.split('\n'):
            if "  " in item:  ## correcting double spaces case
                item = item.replace("  ", " ")
            if sr_prefix in item:
                sr_instances.append(item)
        print(sr_instances)
        if len(sr_instances)==0:
            return type_d, 'No scale found!'
        else:
            sr = sr_instances[0].replace(sr_prefix, "")
            sr = int(sr)

            max_dim = max(self.page.rect.width, self.page.rect.height)
            a0_size = 1189

            fraction =  sr * (a0_size/max_dim)
            return type_d, fraction
    
    def detect_legend(self):
        pix = self.page.get_pixmap(dpi=70)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        img.save('ttest.png')

        self.legend_model.conf = 0.25  # NMS confidence threshold
        self.legend_model.iou = 0.45  # NMS IoU threshold
        results = self.legend_model(img, size=640)
        print('num of legend detections: ', len(results.pred))

        max_conf_idx = -1
        max_conf = 0
        for idx, pred in enumerate(results.pred):
            score = pred[:, 4].cpu().numpy()[0]
            print(score)
            if score > max_conf:
                max_conf = score
                max_conf_idx = idx

        predictions = results.pred[max_conf_idx]
        box = predictions[:, :4].cpu().numpy()[0].astype(int)
        score = predictions[:, 4].cpu().numpy()[0] 
        print('---legend box', box)
        img_viz = np.array(img.copy(), dtype=np.uint8)
        legend_crop = img_viz[box[1]:box[3], box[0]:box[2], :]
        
        self.legend_crop = legend_crop
        self.legend_box = box

        #self.get_vectorgraphics(box, 'legend_only.pdf')
        
        return legend_crop

    



    def detect_hatches(self):
        img = self.legend_crop.copy()
        self.hatch_model.conf = 0.25  # NMS confidence threshold
        self.hatch_model.iou = 0.45  # NMS IoU threshold
        results = self.hatch_model(img, size=1280)

        preds = results.pred[0]
        print('num of hatch detections: ', len(preds))

        hatch_boxes_in_globalcoords = []

        for idx, pred in enumerate(preds):
            pred = pred.cpu().numpy()
            conf = pred[-2]
            cat = int(pred[-1])
            if conf > 0.3:
                box = pred[:4].astype(int)
                # xmin, xmax = box[1], box[3]
                # ymin, ymax = box[0], box[2]
                # print(box)
                color = (0,0,255) if cat==1 else (255,0,0)
                img = bbv.draw_rectangle(img, box, bbox_color=color, is_opaque=True, alpha=0.5)
                

                if cat==0: 
                    box_in_global = [ box[0]+self.legend_box[0], box[1]+self.legend_box[1], 
                                        box[2]+self.legend_box[0], box[3]+self.legend_box[1] ]
                    #paths = self.get_vectorgraphics(box_in_global, self.drawing_filename, str(idx)+'_hatch_only.pdf')
                    
                    #print(len(paths))
    
        return img

    def detect_legend_hatches(self):
        legend_crop = self.detect_legend()
        hatches = self.detect_hatches()
        return legend_crop, hatches

    
    def get_vectorgraphics(self, box, infilename, outfilename):
        cmin,rmin,cmax,rmax = box

        doc = fitz.open(infilename)
        page = doc[0]
    
        W = page.rect.width
        H = page.rect.height
        
        rotation = page.rotation
        page.set_rotation(0)

        paths = page.get_drawings()

        rect = fitz.Rect(rmin, W-cmax, rmax, W-cmin) 

        outpdf = fitz.open()
        outpage = outpdf.new_page(width=H, height=W)
        shape = outpage.new_shape()
        
        legend_paths = []

        for path in paths[::-1][:100000]:

            for item in path["items"]:
                
                if item[0]=="l":
                    if item[1] in rect or item[2] in rect:
                        shape.draw_line(item[1], item[2])
                        legend_paths.append(item)
                elif item[0]=="re":
                    if item[1] in rect:
                        shape.draw_rect(item[1])
                        legend_paths.append(item)
                elif item[0] == "qu":
                    if item[1] in rect:
                        shape.draw_quad(item[1])
                        legend_paths.append(item)
                elif item[0] == "c":  # curve
                    if item[1] in rect or item[2] in rect or item[3] in rect or item[4] in rect:
                        shape.draw_bezier(item[1], item[2], item[3], item[4])
                        legend_paths.append(item)
                else:
                    raise ValueError("unhandled drawing", item)

            shape.finish(
                fill=path["fill"],  # fill color
                color=path["color"],  # line color
                dashes=path["dashes"],  # line dashing
                even_odd=path.get("even_odd", True),  # control color of overlaps
                closePath=path["closePath"],  # whether to connect last and first point
                # lineJoin=path["lineJoin"],  # how line joins should look like
                # lineCap=max(path["lineCap"]),  # how line ends should look like
                width=path["width"],  # line width
                # stroke_opacity=path.get("stroke_opacity", 1),  # same value for both
                # fill_opacity=path.get("fill_opacity", 1),  # opacity parameters
                ) 

        shape.commit()
        
        outpage.set_rotation(rotation)
        #outpdf.save(outfilename)

        return legend_paths
        


        

        

    def segment(self):
        pass

    def count(self):
        pass
        




if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



    #sample_doc = './demo/examples/sample_drawing.pdf'
    sample_doc = './data/samples/AMS13_A-S-002.pdf'
    pipeline = CdrawingAnalyzer(sample_doc, device=device)
    sr = pipeline.calibrate_scale()
    print(sr)
    legend = pipeline.get_legend()
    cv2.imwrite('legend.png', legend)
    hatches = pipeline.detect_hatches()
    cv2.imwrite('hatches.png', hatches)

