import gradio as gr
from gradio_pdf import PDF

import numpy as np
import fitz
import json
import pandas as pd
import cv2
import bbox_visualizer2 as bbv

import sys
sys.path.append('.')
from infer_pipeline import CdrawingAnalyzer

model = CdrawingAnalyzer( device='cpu')

colors = [  (255,165,0), (255,0,255), (127,255,212), (115,147,179), (0,0,255), (255,0,0), (0,255,0), 
            (218,112,214), (46,139,87), ] 

legend_txt = ['ACT 1 - ACOUSTIC CEILING TILES 600X600', 'ACT 2 - ACOUSTIC CEILING TILES 1200X600', 
              'MR GWB - MOISTURE RESISTANT GYPSUM WALL BOARD', 'BFC - BAFFLE CEILING',
              'RF - RAFTS', 'EXPOSED CEILING', 'CEILING TYPE', 'CEILING HEIGHT' ]

def find_sr(drawing):
    doc = fitz.open(drawing)
    page = doc[0]

    text = page.get_text()

    sr_prefix = 'Scale 1 : '

    sr_instances = []

    for item in text.split('\n'):
        if "  " in item:  ## correcting double spaces case
            item = item.replace("  ", " ")
        if sr_prefix in item:
            sr_instances.append(item)
    print(sr_instances)
    if len(sr_instances)==0:
        return 'No scale found!'
    else:
        sr = sr_instances[0].replace(sr_prefix, "")
        sr = int(sr)

        max_dim = max(page.rect.width, page.rect.height)
        a0_size = 1189

        fraction =  sr * (a0_size/max_dim)
        return fraction
        
def find_legend(drawing):
    doc = fitz.open(drawing)
    page = doc[0]

    pix = page.get_pixmap() 
    # pix.save('image.png')

    img = cv2.imread('./results/image.png', 1)


    df = pd.read_csv('./data/labels_box.csv')
    lg = df[df['label_name']=='legend']
    x,y = lg.bbox_x[0], lg.bbox_y[0]
    w,h = lg.bbox_width[0], lg.bbox_height[0]

    crop = img[y:y+h, x:x+w, :]
    cv2.imwrite('./results/legend_crop.png', crop)

    for idx, key in enumerate(['hatch1', 'hatch2', 'hatch3', 'hatch4', 'hatch5', 'hatch6', 'hatch7', 'hatch8']):
      d = df.loc[df['label_name'] == key]

      Hx,Hy = d.iloc[0].bbox_x, d.iloc[0].bbox_y
      Hw,Hh = d.iloc[0].bbox_width, d.iloc[0].bbox_height
      #cv2.rectangle(img, (Hx, Hy), (Hx+Hw, Hy+Hh), colors[idx], 2)

      box = [Hx,Hy, Hx+Hw, Hy+Hh]
      label='hatch'
      color = colors[idx]
      img = bbv.draw_rectangle(img, box, bbox_color=color, is_opaque=True, alpha=0.5)
      img = bbv.add_label(img, label, box, size=12, draw_bg=True,
                     text_bg_color=color, alpha=0.5, text_color=(0,0,255),
                     top=True)

    crop = img[y:y+h, x:x+w, :]
    cv2.imwrite('./results/detections_crop.png', crop)

    return './results/legend_crop.png', './results/detections_crop.png'


def compute_boq(drawing):
  img = cv2.imread('./results/image.png', 1)
  orig_img = img.copy()

  quantities = {}

  anns_file = './data/labels.json'
  with open(anns_file, 'r') as f:
    segs = json.load(f)

  for item in segs['annotations']:
    poly = item['segmentation']
    cat_id = item['category_id']

    # get labelname from category id
    for cat in segs['categories']:
      if cat['id']==cat_id:
        label_name = cat['name']

    ## decoding poly
    idx = int(label_name[-1])-1
    poly = np.array(poly[0], dtype=np.int32).reshape( (-1,2) )
    cv2.fillPoly(img, [poly], color=colors[idx])

    ## entry in quantities
    item_key = legend_txt[idx]
    if not item_key in quantities.keys():
      quantities[item_key] = cv2.contourArea(poly) * 88.20474777448072 * 88.20474777448072 / (1000*1000)
  
  alpha = 0.4
  viz = alpha*orig_img + (1-alpha)*img
  cv2.imwrite('./results/segments.png', np.uint8(viz))


  ## BoQ
  boq_dict = {}
  for key in ['SR', 'Item', 'Quantity(in sq.m.)', 'Remarks']: boq_dict[key]=[] 
  for i, q in enumerate (quantities.keys()):
    boq_dict['SR'].append(i+1)
    boq_dict['Item'].append(q)
    boq_dict['Quantity(in sq.m.)'].append(quantities[q])
    boq_dict['Remarks'].append(" ")

  df = pd.DataFrame.from_dict(boq_dict)
  print(df)

  return './results/segments.png', df
    


with gr.Blocks(title='ConstructionAnalytics') as demo:

    gr.HTML("""<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <div style="
          display: inline-flex;
          gap: 0.8rem;
          font-size: 1.75rem;
          justify-content: center;
          margin-bottom: 10px;
        ">
      <img
        src="https://cdn1.iconfinder.com/data/icons/unigrid-bluetone-maps-travel-vol-2/60/007_074_map_plan_scheme_architectural_apartment_house_construction_building_drafting-512.png"
        alt="Conai" width="64px">
      <h1 style="font-weight: 900; align-items: center; margin-bottom: 7px; margin-top: 20px;">
        Construction Drawing Analytics using AI
      </h1> 
    </div>

    <div><h1 style="font-weight: 900; align-items: center; margin-bottom: 7px; margin-top: 20px;">
        Demo web app
      </h1></div>
    <div>
      <p style="align-items: center; ">
        A Gradio based webapp to test ML models & Pipelines for Construction drawing Analytics
      
    </div>
  </div>"""
    )

    css = """
    #warning {background-color: #FFCCCB} 
    .feedback {font-size: 100px !important;}
    .feedback textarea {font-size: 100px !important;}
    #image {width: 100px; justify-content: center;}
    """


    with gr.Accordion(label="🧭 Instructions:", open=True, elem_id="accordion"):
      with gr.Row(equal_height=True):
          gr.Markdown("""
          - ⭐️ <b>step1: </b>Upload a construction drawing in pdf format. 
          - ⭐️ <b>step2: </b>Click Find Scale button to calibrate the scale. 
          - ⭐️ <b>step3: </b>Click Find Legend button to detect the legend and hatches. 
          - ⭐️ <b>step2: </b>Click Extract BoQ button to segment using the hatches and compute BoQ. 
          """)    

    with gr.Group():
        with gr.Row(equal_height=True):
            with gr.Column(): 
                drawing = PDF(label="Drawing", height=600)

            with gr.Column(): 
                scale_btn = gr.Button("Find Scale")
                with gr.Row():
                  type_out = gr.Textbox(label="Drawing Type")
                  scale_out = gr.Textbox(label="Scale (1 px = -- in mm)")
                  
                scale_btn.click(fn=model.calibrate_scale, inputs=drawing, outputs=[type_out, scale_out], show_progress=True)

                lgnd_btn = gr.Button("Find Legend+Hatches")
                with gr.Row(): 
                  lgnd_out1 = gr.Image(label="Legend", elem_id="image", width=250)
                  lgnd_out2 = gr.Image(label="Hatches", elem_id="image", width=250)
                lgnd_btn.click(fn=model.detect_legend_hatches, inputs=[], outputs=[lgnd_out1, lgnd_out2], show_progress=True)

        with gr.Row():
          boq_btn = gr.Button("Extract BoQ (hardcoded for now)")

        with gr.Row():
          with gr.Column():
            segments = gr.Image(label="Legend-driven-Segments")
          with gr.Column():
            boq_df = gr.Dataframe(label='boq', headers=['SR', 'Item', 'Quantity', 'Remarks'], row_count=(5, 'dynamic'))

        boq_btn.click(fn=compute_boq, inputs=drawing, outputs=[segments, boq_df], show_progress=True)

    with gr.Group():
      examples = ['demo/examples/sample_drawing.pdf']
      gr.Examples(examples=examples, inputs=drawing)


demo.launch(debug=True)