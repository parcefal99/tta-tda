#!/bin/bash

GDOWN="/workspace/miniconda3/envs/ml/bin/gdown"
DATA="/workspace/tta_tda/TDA/dataset"
mkdir -p "$DATA"

gdrive() {
    local id="$1" dest="$2"
    $GDOWN "https://drive.google.com/uc?id=${id}" -O "$dest"
}

ok()   { echo "  [OK] $1"; }
skip() { echo "  [SKIP] $1 already exists"; }
warn() { echo "  [WARN] $1"; }

echo "=========================================="
echo "NOTE: ImageNet (ILSVRC2012) must be downloaded manually."
echo "      Place val/ under $DATA/imagenet/images/val/"
echo "=========================================="

# ---------------------------------------------------------------------------
# classnames.txt
# ---------------------------------------------------------------------------
mkdir -p "$DATA/imagenet/images"
if [ ! -f "$DATA/imagenet/classnames.txt" ]; then
    echo "[classnames.txt] Downloading..."
    gdrive "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF" "$DATA/imagenet/classnames.txt" && ok "classnames.txt"
else
    skip "classnames.txt"
fi

# ---------------------------------------------------------------------------
# Caltech101
# ---------------------------------------------------------------------------
echo ""
echo "[Caltech101]"
mkdir -p "$DATA/caltech-101"
if [ ! -d "$DATA/caltech-101/101_ObjectCategories" ]; then
    wget -q --show-progress -O /tmp/caltech101.zip \
        "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip" && \
    unzip -q /tmp/caltech101.zip -d /tmp/caltech_tmp/ && \
    tar -xzf /tmp/caltech_tmp/caltech-101/101_ObjectCategories.tar.gz -C "$DATA/caltech-101/" && \
    rm -rf /tmp/caltech101.zip /tmp/caltech_tmp && ok "Images extracted"
else
    skip "101_ObjectCategories"
fi
if [ ! -f "$DATA/caltech-101/split_zhou_Caltech101.json" ]; then
    gdrive "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" "$DATA/caltech-101/split_zhou_Caltech101.json" && ok "split JSON"
else
    skip "split_zhou_Caltech101.json"
fi

# ---------------------------------------------------------------------------
# OxfordPets
# ---------------------------------------------------------------------------
echo ""
echo "[OxfordPets]"
mkdir -p "$DATA/oxford_pets"
if [ ! -d "$DATA/oxford_pets/images" ]; then
    wget -q --show-progress -O /tmp/pets_images.tar.gz \
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" && \
    tar -xzf /tmp/pets_images.tar.gz -C "$DATA/oxford_pets/" && \
    rm -f /tmp/pets_images.tar.gz && ok "Images"
else
    skip "images/"
fi
if [ ! -d "$DATA/oxford_pets/annotations" ]; then
    wget -q --show-progress -O /tmp/pets_ann.tar.gz \
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" && \
    tar -xzf /tmp/pets_ann.tar.gz -C "$DATA/oxford_pets/" && \
    rm -f /tmp/pets_ann.tar.gz && ok "Annotations"
else
    skip "annotations/"
fi
if [ ! -f "$DATA/oxford_pets/split_zhou_OxfordPets.json" ]; then
    gdrive "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" "$DATA/oxford_pets/split_zhou_OxfordPets.json" && ok "split JSON"
else
    skip "split_zhou_OxfordPets.json"
fi

# ---------------------------------------------------------------------------
# StanfordCars (Stanford server is often down — skip images gracefully)
# ---------------------------------------------------------------------------
echo ""
echo "[StanfordCars]"
mkdir -p "$DATA/stanford_cars"
if [ ! -d "$DATA/stanford_cars/cars_test" ]; then
    wget -q --show-progress -O /tmp/cars_test.tgz \
        "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz" && \
    tar -xzf /tmp/cars_test.tgz -C "$DATA/stanford_cars/" && \
    rm -f /tmp/cars_test.tgz && ok "Test images" || \
    warn "cars_test.tgz unavailable - download manually from http://ai.stanford.edu/~jkrause/car196/cars_test.tgz"
else
    skip "cars_test/"
fi
if [ ! -f "$DATA/stanford_cars/cars_test_annos_withlabels.mat" ]; then
    wget -q --show-progress -O "$DATA/stanford_cars/cars_test_annos_withlabels.mat" \
        "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat" && ok "Labels" || \
    warn "cars_test_annos_withlabels.mat unavailable - download manually"
else
    skip "cars_test_annos_withlabels.mat"
fi
if [ ! -f "$DATA/stanford_cars/split_zhou_StanfordCars.json" ]; then
    gdrive "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT" "$DATA/stanford_cars/split_zhou_StanfordCars.json" && ok "split JSON"
else
    skip "split_zhou_StanfordCars.json"
fi

# ---------------------------------------------------------------------------
# Flowers102
# ---------------------------------------------------------------------------
echo ""
echo "[Flowers102]"
mkdir -p "$DATA/oxford_flowers"
if [ ! -d "$DATA/oxford_flowers/jpg" ]; then
    wget -q --show-progress -O /tmp/flowers.tgz \
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" && \
    tar -xzf /tmp/flowers.tgz -C "$DATA/oxford_flowers/" && \
    rm -f /tmp/flowers.tgz && ok "Images"
else
    skip "jpg/"
fi
if [ ! -f "$DATA/oxford_flowers/imagelabels.mat" ]; then
    wget -q --show-progress -O "$DATA/oxford_flowers/imagelabels.mat" \
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" && ok "Labels"
else
    skip "imagelabels.mat"
fi
if [ ! -f "$DATA/oxford_flowers/cat_to_name.json" ]; then
    gdrive "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" "$DATA/oxford_flowers/cat_to_name.json" && ok "cat_to_name.json"
else
    skip "cat_to_name.json"
fi
if [ ! -f "$DATA/oxford_flowers/split_zhou_OxfordFlowers.json" ]; then
    gdrive "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" "$DATA/oxford_flowers/split_zhou_OxfordFlowers.json" && ok "split JSON"
else
    skip "split_zhou_OxfordFlowers.json"
fi

# ---------------------------------------------------------------------------
# Food101
# ---------------------------------------------------------------------------
echo ""
echo "[Food101]"
if [ ! -d "$DATA/food-101/images" ]; then
    wget -q --show-progress -O /tmp/food101.tar.gz \
        "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz" && \
    tar -xzf /tmp/food101.tar.gz -C "$DATA/" && \
    rm -f /tmp/food101.tar.gz && ok "Images"
else
    skip "food-101/images/"
fi
if [ ! -f "$DATA/food-101/split_zhou_Food101.json" ]; then
    gdrive "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl" "$DATA/food-101/split_zhou_Food101.json" && ok "split JSON"
else
    skip "split_zhou_Food101.json"
fi

# ---------------------------------------------------------------------------
# FGVCAircraft
# ---------------------------------------------------------------------------
echo ""
echo "[FGVCAircraft]"
if [ ! -d "$DATA/fgvc_aircraft" ]; then
    wget -q --show-progress -O /tmp/fgvc.tar.gz \
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" && \
    mkdir -p /tmp/fgvc_extract && \
    tar -xzf /tmp/fgvc.tar.gz -C /tmp/fgvc_extract/ && \
    mv /tmp/fgvc_extract/fgvc-aircraft-2013b/data "$DATA/fgvc_aircraft" && \
    rm -rf /tmp/fgvc.tar.gz /tmp/fgvc_extract && ok "Data"
else
    skip "fgvc_aircraft/"
fi

# ---------------------------------------------------------------------------
# SUN397
# ---------------------------------------------------------------------------
echo ""
echo "[SUN397]"
mkdir -p "$DATA/sun397"
if [ ! -d "$DATA/sun397/SUN397" ]; then
    wget -q --show-progress -O /tmp/sun397.tar.gz \
        "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz" && \
    tar -xzf /tmp/sun397.tar.gz -C "$DATA/sun397/" && \
    rm -f /tmp/sun397.tar.gz && ok "Images"
else
    skip "SUN397/"
fi
if [ ! -f "$DATA/sun397/split_zhou_SUN397.json" ]; then
    gdrive "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq" "$DATA/sun397/split_zhou_SUN397.json" && ok "split JSON"
else
    skip "split_zhou_SUN397.json"
fi

# ---------------------------------------------------------------------------
# DTD
# ---------------------------------------------------------------------------
echo ""
echo "[DTD]"
if [ ! -d "$DATA/dtd" ]; then
    wget -q --show-progress -O /tmp/dtd.tar.gz \
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" && \
    tar -xzf /tmp/dtd.tar.gz -C "$DATA/" && \
    rm -f /tmp/dtd.tar.gz && ok "Images"
else
    skip "dtd/"
fi
if [ ! -f "$DATA/dtd/split_zhou_DescribableTextures.json" ]; then
    gdrive "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" "$DATA/dtd/split_zhou_DescribableTextures.json" && ok "split JSON"
else
    skip "split_zhou_DescribableTextures.json"
fi

# ---------------------------------------------------------------------------
# EuroSAT
# ---------------------------------------------------------------------------
echo ""
echo "[EuroSAT]"
mkdir -p "$DATA/eurosat"
if [ ! -d "$DATA/eurosat/2750" ]; then
    wget -q --show-progress -O /tmp/eurosat.zip \
        "https://madm.dfki.de/files/sentinel/EuroSAT.zip" && \
    unzip -q /tmp/eurosat.zip -d "$DATA/eurosat/" && \
    if [ -d "$DATA/eurosat/EuroSAT" ] && [ ! -d "$DATA/eurosat/2750" ]; then
        mv "$DATA/eurosat/EuroSAT/"* "$DATA/eurosat/" && rmdir "$DATA/eurosat/EuroSAT"
    fi && \
    rm -f /tmp/eurosat.zip && ok "Images"
else
    skip "2750/"
fi
if [ ! -f "$DATA/eurosat/split_zhou_EuroSAT.json" ]; then
    gdrive "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o" "$DATA/eurosat/split_zhou_EuroSAT.json" && ok "split JSON"
else
    skip "split_zhou_EuroSAT.json"
fi

# ---------------------------------------------------------------------------
# UCF101
# ---------------------------------------------------------------------------
echo ""
echo "[UCF101]"
mkdir -p "$DATA/ucf101"
if [ ! -d "$DATA/ucf101/UCF-101-midframes" ]; then
    gdrive "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" /tmp/ucf101.zip && \
    unzip -q /tmp/ucf101.zip -d "$DATA/ucf101/" && \
    rm -f /tmp/ucf101.zip && ok "Midframes"
else
    skip "UCF-101-midframes/"
fi
if [ ! -f "$DATA/ucf101/split_zhou_UCF101.json" ]; then
    gdrive "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" "$DATA/ucf101/split_zhou_UCF101.json" && ok "split JSON"
else
    skip "split_zhou_UCF101.json"
fi

# ---------------------------------------------------------------------------
# ImageNetV2
# ---------------------------------------------------------------------------
echo ""
echo "[ImageNetV2]"
mkdir -p "$DATA/imagenetv2"
if [ ! -d "$DATA/imagenetv2/imagenetv2-matched-frequency-format-val" ]; then
    wget -q --show-progress -O /tmp/imagenetv2.tar.gz \
        "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz" && \
    tar -xzf /tmp/imagenetv2.tar.gz -C "$DATA/imagenetv2/" && \
    rm -f /tmp/imagenetv2.tar.gz && ok "Images"
else
    skip "imagenetv2-matched-frequency-format-val/"
fi
[ -f "$DATA/imagenet/classnames.txt" ] && \
    cp "$DATA/imagenet/classnames.txt" "$DATA/imagenetv2/" && ok "classnames.txt copied"

# ---------------------------------------------------------------------------
# ImageNet-A
# ---------------------------------------------------------------------------
echo ""
echo "[ImageNet-A]"
mkdir -p "$DATA/imagenet-adversarial"
if [ ! -d "$DATA/imagenet-adversarial/imagenet-a" ]; then
    wget -q --show-progress -O /tmp/imagenet-a.tar \
        "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar" && \
    tar -xf /tmp/imagenet-a.tar -C "$DATA/imagenet-adversarial/" && \
    rm -f /tmp/imagenet-a.tar && ok "Images"
else
    skip "imagenet-a/"
fi
[ -f "$DATA/imagenet/classnames.txt" ] && \
    cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-adversarial/" && ok "classnames.txt copied"

# ---------------------------------------------------------------------------
# ImageNet-R
# ---------------------------------------------------------------------------
echo ""
echo "[ImageNet-R]"
mkdir -p "$DATA/imagenet-rendition"
if [ ! -d "$DATA/imagenet-rendition/imagenet-r" ]; then
    wget -q --show-progress -O /tmp/imagenet-r.tar \
        "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" && \
    tar -xf /tmp/imagenet-r.tar -C "$DATA/imagenet-rendition/" && \
    rm -f /tmp/imagenet-r.tar && ok "Images"
else
    skip "imagenet-r/"
fi
[ -f "$DATA/imagenet/classnames.txt" ] && \
    cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-rendition/" && ok "classnames.txt copied"

# ---------------------------------------------------------------------------
# ImageNet-Sketch
# ---------------------------------------------------------------------------
echo ""
echo "[ImageNet-Sketch]"
mkdir -p "$DATA/imagenet-sketch"
if [ ! -d "$DATA/imagenet-sketch/images" ]; then
    gdrive "1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA" /tmp/imagenet-sketch.zip && \
    unzip -q /tmp/imagenet-sketch.zip -d "$DATA/imagenet-sketch/" && \
    if [ ! -d "$DATA/imagenet-sketch/images" ]; then
        mkdir -p "$DATA/imagenet-sketch/images"
        find "$DATA/imagenet-sketch" -maxdepth 1 -name "n*" -type d \
            -exec mv {} "$DATA/imagenet-sketch/images/" \;
    fi && \
    rm -f /tmp/imagenet-sketch.zip && ok "Images"
else
    skip "images/"
fi
[ -f "$DATA/imagenet/classnames.txt" ] && \
    cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-sketch/" && ok "classnames.txt copied"

echo ""
echo "=========================================="
echo "Done. Summary of dataset directory:"
for d in imagenet caltech-101 oxford_pets stanford_cars oxford_flowers food-101 fgvc_aircraft \
          sun397 dtd eurosat ucf101 imagenetv2 imagenet-adversarial imagenet-rendition imagenet-sketch; do
    if [ -d "$DATA/$d" ]; then
        echo "  [OK]      $d"
    else
        echo "  [MISSING] $d"
    fi
done
echo ""
echo "ImageNet requires manual download: https://image-net.org"
echo "  -> Place ILSVRC2012 val set at: $DATA/imagenet/images/val/"
echo "=========================================="
