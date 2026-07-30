"""اختيار الأدلة بأعلى مكسب معلوماتي متوقَّع (Bayes ساذج).

مصدر الحقيقة الوحيد لهذه الخوارزمية — مُستخرَجة من notebooks/04_smart_selection.ipynb لتفادي تكرارها في
05_finalize.ipynb و06_demo.ipynb. أي تعديل على منطق الاختيار يجب أن يحدث هنا فقط.
"""

import numpy as np


def ecode(item):
    """الكود الأساسي لعنصر دليل خام، بتجاهل قيمة C/M إن وُجدت (E_204_@_V_7 -> E_204)."""
    return item.split("_@_")[0]


def build_nb_presence_table(X_train_full, y_train_full, evidence_columns, n_classes, alpha=1.0):
    """يبني جدول P(code موجود | disease) من مصفوفة train مُرمَّزة، مرة واحدة.

    مُبسَّط عمداً: يعامل كل كود كمتغيّر ثنائي (موجود/غائب) بغض النظر عن نوعه B/C/M — لا يُنمذِج القيمة
    المحدَّدة لأدلة C/M متعددة القيم. تنعيم لابلاس بمعامل alpha لتفادي احتمالات صفرية.

    Returns
    -------
    P_present : ndarray (n_classes, n_codes)
    class_prior : ndarray (n_classes,)
    all_codes : list[str]  مُرتَّبة أبجدياً، بنفس ترتيب أعمدة P_present
    code_to_idx : dict[str, int]
    """
    code_to_colidxs = {}
    for idx, name in enumerate(evidence_columns):
        code_to_colidxs.setdefault(ecode(name), []).append(idx)
    all_codes = sorted(code_to_colidxs.keys())
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    X_csc = X_train_full.tocsc()
    class_counts = np.bincount(y_train_full, minlength=n_classes).astype(float)
    class_prior = class_counts / class_counts.sum()

    P_present = np.zeros((n_classes, len(all_codes)))
    for c in all_codes:
        cols = code_to_colidxs[c]
        presence = np.asarray((X_csc[:, cols].sum(axis=1) > 0)).ravel()
        present_counts = np.bincount(y_train_full, weights=presence.astype(float), minlength=n_classes)
        P_present[:, code_to_idx[c]] = (present_counts + alpha) / (class_counts + 2 * alpha)

    return P_present, class_prior, all_codes, code_to_idx


def select_evidence_eig(
    codes_per_patient, initial_code_per_patient, P_present, class_prior, code_to_idx, k_max, eps=1e-12
):
    """اختيار جشع بأعلى مكسب معلوماتي متوقَّع، دفعياً عبر عدة مرضى معاً.

    في كل خطوة، لكل مرشَّح من الأكواد الفعلية المتبقية لدى كل مريض: يُحسَب **التوقع** على النتيجتين
    الممكنتين (ظهور/غياب الكود) للمعتقد اللاحق الناتج — لا القيمة الفعلية — ويُختار الأقل إنتروبيا متوقَّعة.
    بعد الاختيار فقط تُكشَف القيمة الحقيقية (دائماً "موجود"، لأن المرشَّحين من أكواد المريض الفعلية) ويُحدَّث
    المعتقد بها.

    الاختيار تراكمي: أول k عنصر من `code_sequence[i]` هي نفسها أول k عنصر لأي k_max أكبر — فيمكن استخراج
    نقاط تفتيش متعددة (مثل k=3,5,8,12) من تشغيل واحد بأخذ بادئات القائمة الناتجة، دون إعادة التشغيل.

    Parameters
    ----------
    codes_per_patient : list[list[str]]  كل عنصر: كل الأكواد المميزة المتاحة لهذا المريض (تشمل الأولي)
    initial_code_per_patient : list[str]
    P_present, class_prior, code_to_idx : ناتج build_nb_presence_table (أو محمَّلة من models/nb_evidence_table.npz)
    k_max : عدد الخطوات الإضافية بعد الكود الأولي

    Returns
    -------
    code_sequence : list[list[str]]  الأكواد المُختارة بالترتيب لكل مريض (لا تشمل الكود الأولي)
    """
    n = len(codes_per_patient)
    posterior = np.tile(class_prior, (n, 1)).astype(float)
    for i in range(n):
        posterior[i] = posterior[i] * P_present[:, code_to_idx[initial_code_per_patient[i]]]
        posterior[i] /= posterior[i].sum()

    remaining = [[c for c in codes_per_patient[i] if c != initial_code_per_patient[i]] for i in range(n)]
    code_sequence = [[] for _ in range(n)]

    for _ in range(k_max):
        for i in range(n):
            cands = remaining[i]
            if not cands:
                continue
            cand_idx = np.array([code_to_idx[c] for c in cands])
            post_i = posterior[i]
            p_col = P_present[:, cand_idx]

            p_present_marg = post_i @ p_col
            post_if_present = post_i[:, None] * p_col
            post_if_present = post_if_present / (post_if_present.sum(axis=0, keepdims=True) + eps)
            h_if_present = -np.sum(post_if_present * np.log(post_if_present + eps), axis=0)

            post_if_absent = post_i[:, None] * (1.0 - p_col)
            post_if_absent = post_if_absent / (post_if_absent.sum(axis=0, keepdims=True) + eps)
            h_if_absent = -np.sum(post_if_absent * np.log(post_if_absent + eps), axis=0)

            expected_h = p_present_marg * h_if_present + (1 - p_present_marg) * h_if_absent
            best_local = int(np.argmin(expected_h))
            chosen_code = cands[best_local]

            posterior[i] = post_if_present[:, best_local]
            remaining[i].remove(chosen_code)
            code_sequence[i].append(chosen_code)

    return code_sequence
