# MSGFusion: Multimodal Scene Graph-Guided Infrared and Visible Image Fusion

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.12901-b31b1b.svg)](https://arxiv.org/abs/2509.12901)
[![Project](https://img.shields.io/badge/Project-Page-green)](https://github.com/YourUsername/MSGFusion)


[Guihui Li](mailto:guihuilee@stu.ouc.edu.cn)<sup>1†</sup>, [Bowei Dong](mailto:dbw@stu.ouc.edu.cn)<sup>1†</sup>, [Kaizhi Dong](mailto:dongkaizhi@stu.ouc.edu.cn)<sup>2</sup>, [Jiayi Li](mailto:jiayilee@stu.ouc.edu.cn)<sup>1</sup>, [Haiyong Zheng](mailto:zhenghaiyong@ouc.edu.cn)<sup>1*</sup>

<sup>1</sup>College of Computer Science and Technology, Ocean University of China  
<sup>2</sup>College of Electronic Engineering, Ocean University of China

<sup>†</sup>Equal contribution &nbsp;&nbsp; <sup>*</sup>Corresponding author

---

### 🎯 Core Contribution

**First framework to deeply couple textual conceptual semantics with visual attributes and spatial relationships from infrared and visible images, enabling fusion that simultaneously preserves high-level semantics and low-level details.**

[📄 Paper](#) · [💻 Code (Coming Soon)](#-code-coming-soon) · [📊 Results](#-experimental-results)

</div>

---

## 📑 Table of Contents

<details open>
<summary><b>Click to expand/collapse navigation</b></summary>

- **[📌 Highlights](#-highlights)** - Key features and contributions
- **[🔬 Abstract](#-abstract)** - Research overview
- **[🏗️ Method](#%EF%B8%8F-method)** - Architecture and technical details
  - [Overall Framework](#overall-framework)
  - [Multimodal Scene Graph Representation](#multimodal-scene-graph-representation)
  - [Hierarchical Aggregation](#hierarchical-aggregation)
  - [Scene Graph-Driven Fusion](#scene-graph-driven-fusion)
- **[📊 Experimental Results](#-experimental-results)** - Performance evaluation
  - [Qualitative Comparisons](#qualitative-comparisons)
  - [Quantitative Results](#quantitative-results)
  - [Ablation Studies](#ablation-studies)
  - [Downstream Applications](#downstream-applications)
- **[🚀 Getting Started](#-getting-started)** - Installation and usage
- **[📚 Citation](#-citation)** - BibTeX entry
- **[🙏 Acknowledgments](#-acknowledgments)** - Credits and funding
- **[📧 Contact](#-contact)** - Get in touch

</details>

---

## 📌 Highlights

### Key Innovations

```
🔥 Multimodal Scene Graph Coupling    →  First to deeply integrate text & vision scene graphs for IVIF
📊 Hierarchical Semantic Aggregation   →  Object/Region/Global-level cross-modal alignment
🎯 Graph-Driven Adaptive Fusion        →  Affine modulation guided by structured semantics
⚡ Superior SOTA Performance           →  Best mRank on LLVIP/TNO/RoadScene benchmarks
🔄 Exceptional Generalizability        →  Strong performance on medical fusion & downstream tasks
```

### Performance Highlights

| Dataset | mRank ↓ | Key Metrics | Improvement over SOTA |
|---------|---------|-------------|----------------------|
| **LLVIP** | **2.571** | **Q<sub>abf</sub>=0.620, SSIM=0.596** | **+15.3% mRank gain** |
| **TNO** | **2.857** | **SSIM=0.520, SF=11.540** | **Tied 1st SSIM** |
| **RoadScene** | **3.286** | **SSIM=0.565, MI=3.922** | **Highest MI** |

---

## 🔬 Abstract

Infrared and visible image fusion has garnered considerable attention owing to the strong complementarity of these two modalities in complex, harsh environments. While deep learning-based fusion methods have made remarkable advances in feature extraction, alignment, fusion, and reconstruction, they still depend largely on **low-level visual cues** (texture, contrast) and struggle to capture **high-level semantic information**.

Recent attempts using text as semantic guidance rely on **unstructured descriptions** that neither explicitly model entities, attributes, and relationships nor provide spatial localization, thereby limiting fine-grained fusion performance.

**Our Solution**: We introduce **MSGFusion**, a multimodal scene graph-guided fusion framework that:

✅ Explicitly represents **entities, attributes, and spatial relations** via structured scene graphs from both text and vision  
✅ Synchronously refines **high-level semantics and low-level details** through hierarchical aggregation  
✅ Achieves **superior semantic consistency and structural clarity** via graph-driven fusion

**Extensive experiments** on multiple public benchmarks show MSGFusion significantly outperforms state-of-the-art approaches, particularly in:
- Detail preservation and structural clarity
- Semantic consistency across modalities
- Generalizability to downstream tasks (object detection, segmentation, medical fusion)

---

## 🏗️ Method

### Overall Framework

<div align="center">

<img src="https://github.com/user-attachments/assets/afcdd898-e9ef-44b4-a363-c9fca1afeace" width="100%" alt="MSGFusion Framework"/>

**Figure 1**: Overall architecture of MSGFusion. The framework consists of four key components: **(a) Textual Scene Graph Representation** - constructs conceptual semantic graphs from text descriptions; **(b) Visual Scene Graph Representation** - extracts spatial relationships and attributes from visible images; **(c) Multimodal Scene Graph Hierarchical Aggregation** - aligns and fuses text/vision semantics at object/region/global levels; **(d) Scene Graph-Driven Fusion Module** - generates fused images via affine modulation guided by multimodal scene graph embeddings.

</div>

---

### Multimodal Scene Graph Representation

#### 🔹 Textual Scene Graph (TSG)

**Approach**: Structured semantic extraction from natural language descriptions

**Pipeline**:
```
Text Input → Scene Graph Parser → Semantic Concept Encoder 
    ↓
Object-Attribute GAT → Object-Object Relation GAT → Graph Pooling
    ↓
Textual Scene Graph Embedding (Conceptual Semantics)
```

**Key Features**:
- **Explicit entity-attribute-relation modeling** (vs. unstructured text embeddings)
- **GRU-based phrase encoding** for sequential context
- **Dual-layer graph attention** for hierarchical reasoning

---

#### 🔹 Visual Scene Graph (VSG)

**Approach**: Structured spatial-semantic extraction from visible images

**Pipeline**:
```
Visible Image → Faster R-CNN (Object Detection) → ROI Pooling
    ↓
GRU-based Graph Neural Network (Message Passing)
    ↓
Visual Scene Graph Embedding (Spatial Relations + Attributes)
```

**Key Features**:
- **Candidate region proposals** via Region Proposal Network (RPN)
- **Joint object-relation reasoning** through iterative message passing
- **Context-aware edge features** (union boxes capturing interactions)

---

### Hierarchical Aggregation

**Challenge**: Large granularity gap between text (global concepts) and vision (local pixels)

**Solution**: Multi-level semantic alignment

```
Object-Level:   Top-3 salient objects from vision ←→ Entity nodes from text
Region-Level:   Aggregated region features      ←→ Relation subgraphs
Global-Level:   Whole-image context            ←→ Graph-level summary
```

**Mechanism**:
- **Semantic decomposition** of visual features via self-attention
- **Cross-modal pairing** at each hierarchical level
- **Multi-head attention fusion** with CLS token aggregation

**Output**: Unified multimodal scene graph embedding **E** ∈ ℝ<sup>d</sup>

---

### Scene Graph-Driven Fusion

**Core Idea**: Adaptive affine transformation guided by semantic structure

**Mathematical Formulation**:

$$
\psi_f = \mu(\psi_{ir}) \odot E + \lambda(\psi_{vi})
$$

Where:
- **μ(ψ<sub>ir</sub>)**: Weight term from infrared features (dominant guidance)
- **λ(ψ<sub>vi</sub>)**: Bias term from visible features (complementary details)
- **E**: Multimodal scene graph embedding (semantic constraint)
- **⊙**: Hadamard product (element-wise modulation)

**Advantages**:
- ✅ Preserves thermal saliency from infrared
- ✅ Retains texture/structure from visible
- ✅ Ensures semantic consistency via graph guidance

---

## 📊 Experimental Results

### Qualitative Comparisons

#### LLVIP Dataset (Low-Light Conditions)

<div align="center">

<img src="https://github.com/user-attachments/assets/f2d24504-6bee-43ea-95b4-88e0fc817ce9" width="95%" alt="LLVIP Results"/>

**Figure 2**: Qualitative comparison on LLVIP dataset. **Left to right**: Infrared, Visible, NestFuse, SwinFusion, MUFusion, DAFusion, SpTFuse, IF-FILM, TextFusion, **Ours**. Red boxes highlight superior detail preservation (pedestrian textures, background structures) and thermal saliency retention in our method.

</div>

**Key Observations**:
- 🔴 **NestFuse/DAFusion/IF-FILM**: Over-emphasize infrared, lose visible details
- 🟡 **SwinFusion/TextFusion**: Better balance but insufficient thermal contrast
- 🟢 **Ours**: Optimal modality balance - sharp pedestrian edges + rich background textures

---

#### TNO Dataset (Outdoor Scenes)

<div align="center">

<img src="https://github.com/user-attachments/assets/cd1004cf-c786-42ec-9c67-cdaaa5de0c27" width="95%" alt="TNO Results"/>

**Figure 3**: Cross-dataset generalization on TNO. Our method maintains structural coherence and edge sharpness across diverse degradation conditions without retraining.

</div>

---

#### RoadScene Dataset (Bright Daytime)

<div align="center">

<img src="https://github.com/user-attachments/assets/ec837ce7-e4f8-466b-975a-5c494007ce12" width="95%" alt="RoadScene Results"/>

**Figure 4**: Bright-scene robustness on RoadScene. Avoids over-saturation (common in MUFusion/DAFusion) while enhancing thermal targets (vehicles, pedestrians).

</div>

---

### Quantitative Results

#### LLVIP Dataset Performance

<div align="center">

<img src="https://github.com/user-attachments/assets/ce5b0b2d-feab-411f-9c00-2d99716a85ff" width="80%" alt="Table I"/>

**Table 1**: Quantitative comparison on LLVIP test set. **Bold**: Best; <u>Underline</u>: Second-best. Our method achieves **lowest mRank (2.571)** and leads in **Q<sub>abf</sub>, SSIM, AG, SF**.

</div>

**Detailed Analysis**:
- **Q<sub>abf</sub> = 0.620**: +31.7% over TextFusion (structural preservation)
- **SSIM = 0.596**: Best among all methods (semantic consistency)
- **AG = 7.422**: +47.1% over baseline (edge sharpness)
- **SF = 17.869**: Highest spatial frequency (detail richness)

---

#### TNO Dataset Performance

<div align="center">

<img src="https://github.com/user-attachments/assets/55191857-08bf-4ca6-b1e6-d49e234b641f" width="80%" alt="Table II"/>

**Table 2**: Cross-dataset evaluation on TNO (unseen during training). **mRank = 2.857** (2nd best overall), demonstrating strong generalization capability.

</div>

**Key Strengths**:
- **SSIM = 0.520**: Tied 1st with TextFusion
- **SF = 11.540**: +42.3% over baseline (texture preservation under degradation)
- **Consistent performance** across all metrics (no catastrophic failures)

---

### Ablation Studies

#### Component Contribution Analysis

<div align="center">

<img src="https://github.com/user-attachments/assets/fb6375bb-3590-4de9-9676-6e6f3e7edc4a" width="95%" alt="Ablation Visual"/>

**Figure 5**: Visual ablation study. **Progressive improvements**: Baseline → +TSG (textual scene graph) → +MSGHA (hierarchical aggregation) → +VSG (visual scene graph, full model).

</div>

<div align="center">

<img src="https://github.com/user-attachments/assets/b7931bd1-9dd1-49a2-ad6e-518eff9361c8" width="75%" alt="Ablation Table"/>

**Table 3**: Quantitative ablation on LLVIP. Each component brings consistent gains, with **VSG providing the largest boost** (mRank: 2.571 vs. 3.143 baseline, **+18.2% improvement**).

</div>

**Findings**:
1. **TSG alone**: +4.6% mRank improvement (conceptual semantic guidance)
2. **+MSGHA**: Further +18.2% gain (cross-modal alignment)
3. **+VSG (Full)**: Best overall performance (spatial relation reasoning)

**Conclusion**: All three components are necessary and complementary.

---

#### Loss Function Ablation

<div align="center">
  
<img width="404" height="179" alt="Image" src="https://github.com/user-attachments/assets/61a0f662-530a-4132-9725-bc6f26a88ad3" />

**Figure 6**: Impact of local contrast regularization **L<sub>ctr</sub>**. Red boxes show enhanced edge sharpness (pedestrian contours, vehicle wheels) after adding L<sub>ctr</sub>.

</div>

**Quantitative Impact**:
- **Q<sub>abf</sub>**: +45.9% improvement with L<sub>ctr</sub>
- **VIF**: +51.8% improvement
- **AG**: +68.0% improvement (edge intensity preservation)

**Design Insight**: L<sub>ctr</sub> enforces local structural consistency via standard deviation matching, preventing texture smoothing.

---

### Downstream Applications

#### Object Detection (Pedestrian Detection on LLVIP)

<div align="center">

<img src="https://github.com/user-attachments/assets/e758a295-3ee4-410a-8ea8-6431c8d558e0" width="95%" alt="Detection Results"/>

**Figure 7**: YOLOv11 detection results on fused images. **Blue boxes**: Correct detections; **Red boxes**: Missed targets. Our method achieves **0% miss rate** in challenging occlusion scenarios (pedestrians behind obstacles).

</div>

**Quantitative Results**:

| Method | AP@0.5 | AP@0.7 | AP@0.9 | mAP@[0.5:0.95] |
|--------|--------|--------|--------|----------------|
| TextFusion | 0.836 | 0.716 | 0.154 | 0.508 |
| SpTFuse | 0.842 | 0.745 | **0.189** | 0.537 |
| **Ours** | **0.880** | **0.751** | 0.181 | **0.558** |

**Improvement**: +9.8% mAP over TextFusion, +3.9% over SpTFuse

---

#### Semantic Segmentation (SAM on LLVIP/TNO/RoadScene)

<div align="center">

<img src="https://github.com/user-attachments/assets/cb8402c0-c524-4879-99a4-89598cfc8530" width="48%" alt="Segmentation LLVIP"/>
<img src="https://github.com/user-attachments/assets/789569b8-63e8-47ef-bee1-9c191e8043c7" width="48%" alt="Segmentation TNO"/>

**Figure 8**: Segmentation masks generated by Segment-Anything on fused images. **Left**: LLVIP (low-light); **Right**: TNO (outdoor). Our method produces **cleaner boundaries** and **complete masks** (pedestrians, vehicles, roads).

</div>

**Key Observations**:
- ✅ **Sharper object boundaries** (no mask fragmentation)
- ✅ **Better semantic separability** (pedestrians vs. background)
- ✅ **Cross-domain consistency** (robust across LLVIP/TNO/RoadScene)

---

#### Medical Image Fusion Generalization

<div align="center">
<img src="https://github.com/user-attachments/assets/533dceaf-9ec2-40fc-9060-544b611c07e2" width="95%" alt="Loss Ablation Visual"/>

**Figure 9**: MRI-PET fusion on Harvard Medical dataset (zero-shot transfer). Our method preserves **cortical structures** (MRI) while highlighting **metabolic activity** (PET) without retraining.

</div>

**Quantitative Results**:

| Method | Q<sub>abf</sub> | SSIM | VIF | SF | mRank |
|--------|-----------------|------|-----|----|----|
| TextFusion | 0.274 | 0.296 | 0.532 | 10.548 | 2.200 |
| DAFusion | 0.484 | 0.283 | 0.514 | 16.296 | 2.600 |
| **Ours** | **0.515** | **0.384** | **0.537** | **21.014** | **1.200** |

**Significance**: **Zero-shot generalization** to medical domain validates the universality of scene graph semantic guidance.

---

## 🚀 Getting Started

### 💻 Code (Coming Soon)

### 📦 Installation (Preview)

```bash
# Clone repository (Available Soon)
git clone https://github.com/YourUsername/MSGFusion.git
cd MSGFusion

# Create conda environment
conda create -n msgfusion python=3.8
conda activate msgfusion

# Install dependencies
pip install -r requirements.txt

# Install additional packages for scene graph parsing
pip install spacy
python -m spacy download en_core_web_sm
```

**Requirements** (Expected):
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
opencv-python>=4.5.5
scikit-image>=0.18.0
pyyaml>=6.0
tqdm>=4.62.0
tensorboard>=2.8.0

# Scene Graph Dependencies
spacy>=3.2.0
networkx>=2.6.0
```

---

### 🎯 Quick Start (Preview)

#### Training

```bash
# Train on LLVIP dataset
python scripts/train.py \
    --config configs/msgfusion_llvip.yaml \
    --gpu 0 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4
```

#### Testing

```bash
# Evaluate on LLVIP test set
python scripts/test.py \
    --config configs/msgfusion_llvip.yaml \
    --checkpoint pretrained/msgfusion_llvip.pth \
    --dataset llvip \
    --output_dir results/llvip

# Cross-dataset evaluation on TNO
python scripts/test.py \
    --config configs/msgfusion_tno.yaml \
    --checkpoint pretrained/msgfusion_llvip.pth \
    --dataset tno \
    --output_dir results/tno
```

#### Inference (Single Pair)

```bash
# Fuse a single IR-VI pair with text description
python scripts/inference.py \
    --ir_path examples/ir_001.png \
    --vi_path examples/vi_001.png \
    --text "A person walking on the road at night" \
    --checkpoint pretrained/msgfusion_llvip.pth \
    --output fused_001.png
```

---

### 📥 Pretrained Models (Coming Soon)

| Model | Training Data | Size | Download |
|-------|--------------|------|----------|
| MSGFusion-LLVIP | LLVIP (2000 pairs) | ~150 MB | [Google Drive](#) \| [Baidu Pan](#) |
| MSGFusion-Full | LLVIP+TNO+RoadScene | ~180 MB | [Google Drive](#) \| [Baidu Pan](#) |

---

### 📊 Reproducing Paper Results

**Expected Performance** (after downloading pretrained models):

```bash
# Run full evaluation on LLVIP
python scripts/test.py --config configs/msgfusion_llvip.yaml

# Expected output:
# mRank: 2.571
# Qabf: 0.620
# SSIM: 0.596
# VIF: 0.803
# AG: 7.422
# SF: 17.869
# MI: 2.951
# PSNR: 20.105
```

---

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{li2025msgfusion,
  title={MSGFusion: Multimodal Scene Graph-Guided Infrared and Visible Image Fusion},
  author={Li, Guihui and Dong, Bowei and Dong, Kaizhi and Li, Jiayi and Zheng, Haiyong},
  journal={arXiv preprint arXiv:2509.12901},
  year={2025}
}
```

**Note**: Citation will be updated upon journal acceptance.

---

## 🙏 Acknowledgments

### Funding Support

This work was supported by:
- National Natural Science Foundation of China (Grant No. XXXXXXX)
- Shandong Provincial Natural Science Foundation (Grant No. XXXXXXX)
- Ocean University of China Research Fund

### Technical Acknowledgments

We thank the authors of the following works for open-sourcing their code:
- [DenseFuse](https://github.com/hli1221/densefuse-pytorch) - Baseline fusion network
- [Faster R-CNN](https://github.com/pytorch/vision) - Object detection backbone
- [Scene Graph Parsing](https://github.com/vacancy/SceneGraphParser) - Text-to-graph conversion
- [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP) - Low-light paired data

### Competing Methods

We compared with the following state-of-the-art methods:
- NestFuse (TIM 2020), SwinFusion (JAS 2022), MUFusion (IF 2023)
- DAFusion (IF 2025), SpTFuse (PR 2025), IF-FILM (arXiv 2024)
- TextFusion (IF 2025)

---

## 📧 Contact

**Corresponding Author**: Haiyong Zheng  
📧 Email: zhenghaiyong@ouc.edu.cn  
🏫 Affiliation: College of Computer Science and Technology, Ocean University of China

**First Authors**:  
Guihui Li (guihuilee@stu.ouc.edu.cn)  
Bowei Dong (dbw@stu.ouc.edu.cn)

**For questions**:
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/YourUsername/MSGFusion/issues) (Available Soon)
- 💬 **Research Discussion**: Email the corresponding author
- 🤝 **Collaboration**: Contact via institutional email

---

<div align="center">

### ⭐ If you find this work helpful, please star this repository! ⭐

![GitHub stars](https://img.shields.io/github/stars/YourUsername/MSGFusion?style=social)
![GitHub forks](https://img.shields.io/github/forks/YourUsername/MSGFusion?style=social)

---

**Advancing Infrared-Visible Image Fusion with Structured Semantic Guidance**

**Last Updated**: January 2025 | **Status**: Under Review at IEEE TMM

[⬆️ Back to Top](#msgfusion-multimodal-scene-graph-guided-infrared-and-visible-image-fusion)

</div>

---

## 📌 Related Work

### Our Other Publications

- [Coming Soon] More works on multimodal fusion and scene understanding

### Recommended Reading

**Infrared-Visible Fusion**:
- DenseFuse (TIP 2018) - Autoencoder-based fusion
- FusionGAN (IF 2019) - GAN-based fusion
- SwinFusion (JAS 2022) - Transformer-based fusion

**Scene Graph Generation**:
- Neural Motifs (CVPR 2018) - Context modeling
- Unbiased SGG (CVPR 2020) - Debiasing strategies
- Panoptic Scene Graph (ECCV 2022) - Panoptic-level reasoning

**Vision-Language Models**:
- CLIP (ICML 2021) - Contrastive pretraining
- Structure-CLIP (AAAI 2024) - Scene graph integration

---

**🎯 Future Directions**

We are actively working on:
- [ ] Real-time fusion (>30 FPS on edge devices)
- [ ] Unified model for multi-task fusion (IVIF + MEF + MFF)
- [ ] Open-vocabulary scene graph generation
- [ ] Interactive fusion with user-specified scene graphs

**Stay tuned for updates!** 🚀
