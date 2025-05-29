from timm import create_model

model = create_model(
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    pretrained=True,
    num_classes=1000,
    cache_dir="./.cache"
)
