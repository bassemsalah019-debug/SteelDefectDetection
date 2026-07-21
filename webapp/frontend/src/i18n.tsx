import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

type Lang = "en" | "ar";

const DICT: Record<Lang, Record<string, string>> = {
  en: {
    brand: "SteelVision",
    tagline: "Steel surface defect inspection",
    dashboard: "Dashboard", new_inspection: "New Inspection", history: "History",
    logout: "Log out", login: "Log in", signup: "Sign up",
    email: "Email", password: "Password", full_name: "Full name",
    welcome_back: "Welcome back", create_account: "Create your account",
    no_account: "No account?", have_account: "Already have an account?",
    inspections: "Inspections", images: "Images", defects: "Defects", avg_per_image: "Avg / image",
    defects_by_class: "Defects by class", activity_14d: "Activity (14 days)", recent: "Recent inspections",
    mode: "Mode", adaptive: "Adaptive", fixed: "Fixed", inference_size: "Inference size",
    confidence: "Confidence", title: "Title", drop_images: "Drop steel images here or click to browse",
    run_inspection: "Run inspection", analyzing: "Analyzing…", selected: "selected",
    no_inspections: "No inspections yet", start_first: "Run your first inspection to see it here.",
    view: "View", delete: "Delete", original: "Original", detections: "Detections", eigencam: "Eigen-CAM",
    per_class: "Per-class summary", signals: "Adaptive signals", brightness: "Brightness", quality: "Quality",
    report: "Inspection report", generate_report: "Generate report", generating: "Generating…",
    no_defects: "No defects detected", defects_found: "defects found", back: "Back",
    created: "Created", search: "Search", all_modes: "All modes",
  },
  ar: {
    brand: "ستيل فيجن",
    tagline: "فحص عيوب سطح الفولاذ",
    dashboard: "لوحة المعلومات", new_inspection: "فحص جديد", history: "السجل",
    logout: "تسجيل الخروج", login: "تسجيل الدخول", signup: "إنشاء حساب",
    email: "البريد الإلكتروني", password: "كلمة المرور", full_name: "الاسم الكامل",
    welcome_back: "مرحبًا بعودتك", create_account: "أنشئ حسابك",
    no_account: "لا تملك حسابًا؟", have_account: "لديك حساب بالفعل؟",
    inspections: "عمليات الفحص", images: "الصور", defects: "العيوب", avg_per_image: "متوسط / صورة",
    defects_by_class: "العيوب حسب الفئة", activity_14d: "النشاط (14 يومًا)", recent: "أحدث الفحوصات",
    mode: "الوضع", adaptive: "تكيّفي", fixed: "ثابت", inference_size: "حجم الاستدلال",
    confidence: "الثقة", title: "العنوان", drop_images: "أفلِت صور الفولاذ هنا أو انقر للتصفح",
    run_inspection: "تشغيل الفحص", analyzing: "جارٍ التحليل…", selected: "محدد",
    no_inspections: "لا توجد فحوصات بعد", start_first: "شغّل أول فحص لرؤيته هنا.",
    view: "عرض", delete: "حذف", original: "الأصلية", detections: "الاكتشافات", eigencam: "خريطة الانتباه",
    per_class: "ملخص حسب الفئة", signals: "إشارات التكيّف", brightness: "السطوع", quality: "الجودة",
    report: "تقرير الفحص", generate_report: "إنشاء تقرير", generating: "جارٍ الإنشاء…",
    no_defects: "لم تُكتشف عيوب", defects_found: "عيوب مكتشفة", back: "رجوع",
    created: "أُنشئ", search: "بحث", all_modes: "كل الأوضاع",
  },
};

interface I18n { lang: Lang; dir: "ltr" | "rtl"; t: (k: string) => string; setLang: (l: Lang) => void; }
const Ctx = createContext<I18n>(null as unknown as I18n);
export const useI18n = () => useContext(Ctx);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem("sv_lang") as Lang) || "en");
  const dir = lang === "ar" ? "rtl" : "ltr";
  useEffect(() => {
    document.documentElement.dir = dir;
    document.documentElement.lang = lang;
    localStorage.setItem("sv_lang", lang);
  }, [lang, dir]);
  const t = (k: string) => DICT[lang][k] ?? DICT.en[k] ?? k;
  return <Ctx.Provider value={{ lang, dir, t, setLang }}>{children}</Ctx.Provider>;
}
