import sys
print(sys.path)
sys.path.append("coco-caption")
print(sys.path)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap