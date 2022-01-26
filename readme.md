# Oxford AI&ML Course

- Required Libraries

## Session 0: Python Re-fresh

```bash
import math
import random
import datatime
```

## Session 1: Data Cleansing

```bash
import pandas
import numpy
import seaborn as sns
import matplotlib
import missingno    # conda install -c conda-forge missingno
```

## Session 2: End-to-end example of supervised learning

```bash
import sklearn      # conda install scikit-learn
import pickle
```

## Session 3a: Clustering & 3b: Dimensionality reduction

```bash
import mpl_toolkits
import random as rand
```

## Session 4: Gradient Descent

## Session 5: Polynomial Regression and ROC

```bash
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
boston_dataset = load_boston()
```

## Session 6: Trees and Ensemble

```bash
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
```

- 额外的命令

```bash
conda install -c conda-forge graphviz
```

## Session 7: Gaussian Mixture Models

## Session 8: Natural Language Processing

### 8.1 Wordcloud

```bash
from wordcloud import WordCloud, STOPWORDS  # conda install -c conda-forge wordcloud
import matplotlib.pyplot as plt
```

### 8.2 Natural Language Toolkit (NLTK)

```bash
import nltk
from nltk.corpus import stopwords
from nltk.corpus.reader import tagged
```

### 8.3 Vader Sentiment Analyzer

```bash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# conda install -c conda-forge vadersentiment
```

### 8.4 Other libraries used in the Natural Language Workshop

```bash
from collections import OrderedDict
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
```
