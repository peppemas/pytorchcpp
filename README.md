# Simple C++ inference example using opencv + torch

### Requirements
* Use tools/export_model.py to extract the resnet18.pt
* The resnet18_labels.txt was downloaded from the internet
* Ensure to install opencv and torch development libraries on Linux

### Compilation

```bash
  cd pytorchcpp
  mkdir build
  cd build
  cmake ..
  make
```

### Launch Application

```
pytorchcpp ../assets/resnet18.pt ../assets/resnet18_labels.txt ../assets/shark.jpg
```

### Copyright Notice

All images in "assets" folder are downloaded from **Stable Diffusion Web**
