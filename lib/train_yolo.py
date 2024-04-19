### how to train yolov9 model on custom data


#step 1:

### dataset folder organization
'''
./data/
--- data.yaml
        train: ./data/hatch_detection_data/train/images
        val: ./data/hatch_detection_data/valid/images

        nc: 2
        names: ['hatch', 'label']
--- train
    ---images
    ---labels
--- valid
    --- images
    --- labels
'''

## step 2: training
python train_dual.py --workers 8 --device 0 --batch 2 --data data/hatch_detection_data/data.yaml --img 1280 --cfg models/detect/yolov9-e.yaml --weights ../../weights/legend_detection_best.pt --name yolov9eHatch --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 15

## step 3: detect
python detect_dual.py