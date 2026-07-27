# مانيفست بيانات DDXPlus (`data/raw/`)

## المصدر

- **الداتاسِت**: DDXPlus (English)
- **Figshare article ID**: `22687585`
- **API**: `https://api.figshare.com/v2/articles/22687585`
- **التوثيق الكامل**: <https://github.com/mila-iqia/ddxplus>
- **الترخيص**: CC-BY (كما ورد في وصف المقالة على Figshare)
- **تاريخ التحميل**: 2026-07-27

## الملفات

| الملف | الحجم | SHA-256 |
|---|---|---|
| `release_conditions.json` | 21 KB | `56edf4682d8e86a4209fc3a33932e50ce03d1cc1fecc326c26d5ef8fa5c24890` |
| `release_evidences.json` | 118 KB | `281c78b044ae60514e28ecc9052d08a8225dd3e2db3f00f8788b99c47b05e0c0` |
| `release_train_patients.csv` | 640 MB | `93933e9a9a7a00d618a931deda7767485af299fe4635bf189a7c63b510e8b172` |
| `release_validate_patients.csv` | 84 MB | `b84733533bff01daa1d47d27a0cd4d684bb54a0fc20d80aaee1250ff8ff989ed` |
| `release_test_patients.csv` | 85 MB | `f7ac3eae934c85780fc9b109a6cac5771540619a02690bee6fc0ab402baec186` |

> ملاحظة: أسماء الملفات داخل أرشيفات `patients` الأصلية على Figshare (`release_*_patients.zip`) لا تحمل
> امتداد `.csv` — أُعيدت تسميتها هنا فقط بعد فك الضغط لتطابق ما يتوقعه `notebooks/01_explore.ipynb`.
> المحتوى (الأعمدة والقيم) لم يُعدَّل إطلاقاً.

## إعادة الإنتاج

```bash
# JSON files
curl -L -o data/raw/release_evidences.json  https://ndownloader.figshare.com/files/40278013
curl -L -o data/raw/release_conditions.json https://ndownloader.figshare.com/files/62561569

# Patients CSVs (داخل أرشيفات zip)
curl -L -o data/raw/release_train_patients.zip    https://ndownloader.figshare.com/files/40278019
curl -L -o data/raw/release_validate_patients.zip https://ndownloader.figshare.com/files/40278022
curl -L -o data/raw/release_test_patients.zip     https://ndownloader.figshare.com/files/40278016
# ثم فك الضغط وإعادة تسمية الملف الناتج إلى release_{split}_patients.csv
```

للتحقق من سلامة الملفات بعد أي تحميل جديد، احسب الـhash وقارنه يدوياً بالجدول أعلاه:

```bash
sha256sum data/raw/release_conditions.json data/raw/release_evidences.json \
  data/raw/release_train_patients.csv data/raw/release_validate_patients.csv \
  data/raw/release_test_patients.csv
```
