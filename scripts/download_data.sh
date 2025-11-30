mkdir -p data/coco
cd data/coco

# 1) Download train images
curl -O http://images.cocodataset.org/zips/train2017.zip

# 2) Download val images (optional but useful later)
curl -O http://images.cocodataset.org/zips/val2017.zip

# 3) Download annotations
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 4) Unzip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Optionally remove zips to save space
# rm train2017.zip val2017.zip annotations_trainval2017.zip

cd ../..