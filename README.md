IDEA
====

IDEA clustering framework

## ENVIRONMENT
Python 3

## REQUIREMENT
scikit-learn (https://scikit-learn.org/stable/install.html)
PyMetis (https://pypi.org/project/PyMetis/)
```bash
pip install scikit-learn
pip install pybind11
pip install pymetis
```

## EXAMPLE
```bash
python IDEA.py data/A01.data.points 15 -dataType datapoints -o A01.cluster -o2 A01.linkage -o3 A01.graph
python IDEA.py data/A02.data.points 31 -dataType datapoints -o A02.cluster -o2 A02.linkage -o3 A02.graph
python IDEA.py data/A03.data.points 35 -dataType datapoints -o A03.cluster -o2 A03.linkage -o3 A03.graph
python IDEA.py data/A04.data.points 15 -dataType datapoints -o A04.cluster -o2 A04.linkage -o3 A04.graph
python IDEA.py data/A05.data.points 15 -dataType datapoints -o A05.cluster -o2 A05.linkage -o3 A05.graph
python IDEA.py data/A06.data.points 15 -dataType datapoints -o A06.cluster -o2 A06.linkage -o3 A06.graph
python IDEA.py data/B01.data.points 3 -dataType datapoints -o B01.cluster -o2 B01.linkage -o3 B01.graph
python IDEA.py data/B02.data.points 6 -dataType datapoints -o B02.cluster -o2 B02.linkage -o3 B02.graph
python IDEA.py data/B03.data.points 9 -dataType datapoints -o B03.cluster -o2 B03.linkage -o3 B03.graph
python IDEA.py data/B04.data.points 8 -dataType datapoints -o B04.cluster -o2 B04.linkage -o3 B04.graph
python IDEA.py data/B05.data.points 10 -dataType datapoints -o B05.cluster -o2 B05.linkage -o3 B05.graph
python IDEA.py data/B06.data.points 9 -dataType datapoints -o B06.cluster -o2 B06.linkage -o3 B06.graph
python IDEA.py data/C01.data.points 7 -dataType datapoints -o C01.cluster -o2 C01.linkage -o3 C01.graph
python IDEA.py data/C02.data.points 6 -dataType datapoints -o C02.cluster -o2 C02.linkage -o3 C02.graph
python IDEA.py data/C03.data.points 9 -dataType datapoints -o C03.cluster -o2 C03.linkage -o3 C03.graph
python IDEA.py data/C04.data.points 8 -dataType datapoints -o C04.cluster -o2 C04.linkage -o3 C04.graph
python IDEA.py data/C05.data.points 10 -dataType datapoints -o C05.cluster -o2 C05.linkage -o3 C05.graph
python IDEA.py data/C06.data.points 9 -dataType datapoints -o C06.cluster -o2 C06.linkage -o3 C06.graph
python IDEA.py data/D01.data.dist.fullmatrix 11 -dataType dissimilarity -o D01.cluster -o2 D01.linkage -o3 D01.graph
python IDEA.py data/D02.data.dist.sparsematrix 5 -dataType dissimilarity -o D02.cluster -o2 D02.linkage -o3 D02.graph
```
