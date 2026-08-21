# خطوات النشر على PythonAnywhere | Deployment Steps
=====================================================

الهدف: تشغيل المشروع كامل (واجهة + RAG + Gemini) على رابط عام:
**https://dawoods1.pythonanywhere.com**

الملفات المطلوبة للنشر (موجودة في هذا المجلد deploy/):
- `cloud_server.py` — السيرفر السحابي الخفيف (بدون torch)
- `vectors_gemini.jsonl` — فهرس المتجهات المشفّر بـGemini
- `../pro1.html` — الواجهة (تُقدَّم من نفس السيرفر)

---

## 1) ارفع التحديثات إلى GitHub أولًا
ارفع هذه الملفات إلى مستودعك `dawood3m1/EDU-Ai-Project-new`:
- `pro1/pro1.html` (النسخة المحدثة بدعم API_BASE)
- `pro1/deploy/cloud_server.py`
- `pro1/deploy/reindex_gemini.py`
- `pro1/deploy/vectors_gemini.jsonl`
- `pro1/deploy/DEPLOY-STEPS.md`

## 2) اسحب المستودع داخل PythonAnywhere
افتح **Consoles** ← شغّل **Bash** console ونفّذ:
```bash
cd ~
git clone https://github.com/dawood3m1/EDU-Ai-Project-new.git
```

## 3) تأكد من المكتبات (عادة مثبتة مسبقًا)
في نفس الـBash:
```bash
python3 -c "import flask, requests; print('OK')"
```
إن ظهر خطأ: `pip3 install --user flask requests`

## 4) أنشئ تطبيق الويب
- تبويب **Web** ← **Add a new web app**
- Next ← اختر **Manual configuration** ← اختر **Python 3.10** ← Create

## 5) اربط الـWSGI بالسيرفر
من صفحة Web، اضغط على رابط ملف الـWSGI
(`/var/www/dawoods1_pythonanywhere_com_wsgi.py`) واستبدل محتواه كله بهذا:

```python
import os
import sys

# المفتاح يوضع هنا فقط — هذا الملف خاص ولا يُرفع على GitHub أبداً
os.environ["GOOGLE_API_KEY"] = "الصق_مفتاحك_هنا"

sys.path.insert(0, "/home/dawoods1/EDU-Ai-Project-new/pro1/deploy")

from cloud_server import application as application
```

احفظ الملف.

## 6) أعد التشغيل وجرّب
- ارجع لتبويب **Web** ← اضغط الزر الأخضر **Reload**
- افتح: **https://dawoods1.pythonanywhere.com**
- جرّب سؤالًا دراسيًا وتأكد من ظهور مصدر الإجابة (صفحة الكتاب)

---

## ملاحظات مهمة
- **لا تضع المفتاح في أي ملف يُرفع على GitHub** — فقط في ملف WSGI الخاص بحسابك.
- الحساب المجاني يحتاج ضغطة "Run until" كل 3 أشهر من تبويب Web (تنبيه يظهر لك).
- عند أي تحديث للمشروع لاحقًا: `cd ~/EDU-Ai-Project-new && git pull` ثم Reload.
- لو ظهر خطأ، راجع سجل الأخطاء من صفحة Web (رابط Error log).
