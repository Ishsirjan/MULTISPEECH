# 🎙️ MultiSpeech: Multimodal Detection of Speech Sound Disorders from Ultrasound & Audio

This project builds a **deep learning pipeline** that classifies speech data into three categories:

- **Typically Developing (Control)** – UXTD
- **Speech Sound Disorders (SSD)** – UXSSD
- **Clinical/Varied Cases** – UPX

We use both **ultrasound tongue images (.ult)** and **audio recordings (.wav)**, leveraging **multimodal learning** to improve classification performance.

---

## 📂 Dataset: [UltraSuite](https://ultrasuite.github.io/)

We use data from the UltraSuite repository, which contains synchronized:

- `.ult` — raw ultrasound images of the tongue during speech
- `.wav` — audio recordings
- `.txt` — utterance prompts
- `.param` — ultrasound metadata

### 🧠 Classes Used:
| Class | Dataset | Label |
|-------|---------|-------|
| Control | UXTD | 0 |
| Speech Sound Disorder | UXSSD | 1 |
| Clinical Mixed | UPX | 2 |

We sampled from **5 speakers per dataset** for a balanced training set.

> **To use this project**, download the data directly via [UltraSuite](https://ultrasuite.github.io/download_win/) using `rsync` or their zipped samples.

---

## 🧠 Why Multimodal?

Ultrasound captures **articulatory movement**, and audio captures **acoustic output**. Combining both offers richer signals, improving generalization for speech diagnostics.

---

## 🔁 Pipeline Flowchart

[ Ultrasound (.ult) ]        [ Audio (.wav) ]
          ↓                         ↓
  Simulated Image Frame       MFCC Extraction
          ↓                         ↓
       CNN Layer              Feedforward Network
               ↘             ↙
               [ Concatenation ]
                      ↓
               [ Classifier ]
                      ↓
     Output → [Control | SSD | Clinical]

---

## 🧪 Results

On ~4,100 multimodal samples:

- **Best Accuracy**: 86%
- **Balanced Accuracy** (with 5 speakers each): ~79%
- **Model**: CNN for image + Dense layers for audio features

Example confusion matrix:
```
[ [TP_0, FP_0, FN_0],
  [FP_1, TP_1, FN_1],
  [FP_2, FN_2, TP_2] ]
```

---

## 🚀 Quickstart

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**
   - Use `rsync` or download from: https://ultrasuite.github.io/download_win/

3. **Run Training**
   ```bash
   python main.py
   ```

---

## 📁 Folder Structure

```
.
├── main.py               # Main training + evaluation script
├── data_loader.py        # Custom PyTorch Dataset
├── requirements.txt
├── README.md
```

---

## 🧠 What We Did

- Designed a **CNN + Dense multimodal architecture**
- Simulated `.ult` files as grayscale images
- Extracted **MFCCs from `.wav`** files using librosa
- Performed **random split** with ~80/20 train/test
- Balanced samples across classes (UXTD, UXSSD, UPX)
- Evaluated using accuracy, precision, recall, F1-score, confusion matrix

---

## 🔬 Why This Matters

- Can aid in **early screening** of speech disorders
- Offers a **non-invasive, low-cost diagnostic aid**
- Potential for use in **low-resource clinics** or tele-therapy setups

---



## 📜 License

MIT License — Open for research and non-commercial use.