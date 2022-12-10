# Some common data proprocessing functions


Categorical Variables to Numerical Labels
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for k in dataset1.keys(): dataset1[k] = encoder.fit_transform(dataset1[k])
# note that a whole list of keys can also be passed in
```

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
```python
```