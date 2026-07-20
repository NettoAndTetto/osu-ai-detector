# Mapper Classifier

Try the model [here](https://colab.research.google.com/github/OliBomby/Mapperatorinator/blob/main/colab/classifier_classify.ipynb).

Mapper Classifier is a model that predicts which osu! standard ranked mapper mapped a given beatmap.

This model is built using transfer learning on the Mapperatorinator V22 model.
It achieves a top-1 validation accuracy of 12.5% on a random sample of ranked beatmaps and recognizes 3,731 unique mappers.
To make its predictions, the model analyzes an 8-second segment of beatmap.

The purpose of this classifier is actually to calculate high-level feature vectors for beatmaps, which can be used to calculate the similarity between generated beatmaps and real beatmaps.
This is a technique often used to assess the quality of image generation models with the [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500).
However, in my testing I found that the computed FID scores for beatmap generation models were not very close to the actual quality of the generated beatmaps.
This classifier might not be able to recognize all the necessary features to accurately assess the quality of a beatmap, but it's a start.

## Usage

Run `classify.py` with the path to the beatmap you want to classify and the time in seconds of the segment you want to use to classify the beatmap.
```shell
python classify.py beatmap_path="'...\Songs\1790119 THE ORAL CIGARETTES - ReI\THE ORAL CIGARETTES - ReI (Sotarks) [Cataclysm.].osu'" time=60
```

```
Mapper: Sotarks (4452992) with confidence: 9.760356903076172
Mapper: Sajinn (13513687) with confidence: 6.975161075592041
Mapper: kowari (5404892) with confidence: 6.800069332122803
Mapper: Haruto (3772301) with confidence: 6.077754020690918
Mapper: Kalibe (3376777) with confidence: 5.894346237182617
Mapper: iljaaz (8501291) with confidence: 5.873990535736084
Mapper: tomadoi (5712451) with confidence: 5.817874431610107
Mapper: Nao Tomori (5364763) with confidence: 5.144880294799805
Mapper: Kujinn (3723568) with confidence: 5.082106590270996
...
```