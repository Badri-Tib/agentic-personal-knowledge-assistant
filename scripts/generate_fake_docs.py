#!/usr/bin/env python3
"""
generate_fake_docs.py -- Generate anonymised example PDFs for the demo.

Creates three realistic but entirely fictitious documents in data/examples/
based on the character Subaru Natsuki from Re:Zero.

  - cv_subaru_natsuki.pdf          (2 pages)
  - releve_notes_m1.pdf            (1 page)
  - titre_sejour_exemple.pdf       (1 page)

Run once:
    python scripts/generate_fake_docs.py
"""

from pathlib import Path
import fitz  # pymupdf

OUT_DIR = Path(__file__).parent.parent / "data" / "examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Colour palette -----------------------------------------------------------
BLACK = (0.0, 0.0, 0.0)
DARK  = (0.15, 0.15, 0.15)
GREY  = (0.45, 0.45, 0.45)
BLUE  = (0.10, 0.27, 0.60)
LIGHT = (0.92, 0.94, 0.97)

# A4 in points
W, H   = 595, 842
MARGIN = 50


# -- Low-level helpers ---------------------------------------------------------

def txt(page, x, y, text, size=11, font="helv", color=BLACK):
    page.insert_text(fitz.Point(x, y + size), text,
                     fontname=font, fontsize=size, color=color)


def box(page, x0, y0, x1, y1, text, size=11, font="helv",
        color=BLACK, align=0):
    page.insert_textbox(
        fitz.Rect(x0, y0, x1, y1 + size),
        text, fontname=font, fontsize=size, color=color, align=align,
    )


def hline(page, y, x0=MARGIN, x1=W - MARGIN, color=BLUE, width=0.8):
    page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y),
                   color=color, width=width)


def section(page, y, title):
    txt(page, MARGIN, y, title.upper(), size=10, font="hebo", color=BLUE)
    hline(page, y + 14, width=0.5)
    return y + 22


def fill_rect(page, x0, y0, x1, y1, color=LIGHT):
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=None, fill=color)


# -- Document 1 -- CV ---------------------------------------------------------

def make_cv():
    doc = fitz.open()

    # -- Page 1 ----------------------------------------------------------------
    p = doc.new_page(width=W, height=H)

    fill_rect(p, 0, 0, W, 115, color=BLUE)
    p.insert_text(fitz.Point(MARGIN, 50), "NATSUKI Subaru",
                  fontname="hebo", fontsize=24, color=(1, 1, 1))
    p.insert_text(fitz.Point(MARGIN, 72), "Ingenieur IA / Machine Learning",
                  fontname="helv", fontsize=13, color=(0.85, 0.90, 1.0))
    contact = ("subaru.natsuki@email.com  |  +33 6 42 00 00 01  |  "
               "Paris, France  |  linkedin.com/in/subarunatsuki")
    p.insert_text(fitz.Point(MARGIN, 95), contact,
                  fontname="helv", fontsize=9, color=(0.80, 0.88, 1.0))

    y = 130

    # Profil
    y = section(p, y, "Profil")
    profil = (
        "Ingenieur IA passionne par le NLP, le RAG et l'apprentissage par renforcement. "
        "Diplome d'un double Master (Big Data & IA + Computer Vision) a l'Universite Paris 8. "
        "Resilience exceptionnelle face aux echecs -- capable de relancer indefiniment une "
        "experience jusqu'a convergence. Cherche un poste CDI d'AI/ML Engineer a Paris."
    )
    box(p, MARGIN, y, W - MARGIN, y + 55, profil, size=10, color=DARK)
    y += 65

    # Experiences
    y = section(p, y, "Experiences professionnelles")

    txt(p, MARGIN, y, "Data Scientist -- Roswaal Technologies", size=11, font="hebo")
    txt(p, W - MARGIN - 120, y, "Sept. 2023 - Aout 2024", size=9, color=GREY)
    y += 16
    for b in [
        "- Developpement d'un agent RAG pour la base de connaissances interne (LangGraph + ChromaDB).",
        "- Fine-tuning de modeles de classification (BERT, CamemBERT) : +9 pts F1.",
        "- Mise en production via FastAPI / Docker sur GCP.",
        "- Evaluation des reponses LLM avec RAGAS (faithfulness, context recall).",
    ]:
        txt(p, MARGIN + 8, y, b, size=10, color=DARK)
        y += 14
    y += 6

    txt(p, MARGIN, y, "Stagiaire Machine Learning -- Emilia Research Lab", size=11, font="hebo")
    txt(p, W - MARGIN - 120, y, "Avr. 2022 - Aout 2022", size=9, color=GREY)
    y += 16
    for b in [
        "- Entrainement d'un modele de detection d'objets (YOLOv8) sur donnees propriete.",
        "- Annotation de dataset (Roboflow), augmentation, evaluation mAP@50.",
        "- Reduction du taux d'erreur de 18 % vs. baseline.",
    ]:
        txt(p, MARGIN + 8, y, b, size=10, color=DARK)
        y += 14
    y += 6

    txt(p, MARGIN, y, "Assistant de recherche -- LIPN, Universite Paris 13", size=11, font="hebo")
    txt(p, W - MARGIN - 120, y, "Oct. 2021 - Mars 2022", size=9, color=GREY)
    y += 16
    for b in [
        "- Reproduction de l'architecture TopicBERT pour la modelisation thematique.",
        "- Evaluation des metriques coherence et diversity sur corpus Wikipedia FR.",
    ]:
        txt(p, MARGIN + 8, y, b, size=10, color=DARK)
        y += 14
    y += 6

    # Formation
    y = section(p, y, "Formation")

    txt(p, MARGIN, y, "Master 2 -- Big Data & Intelligence Artificielle", size=11, font="hebo")
    txt(p, W - MARGIN - 80, y, "2023 - 2024", size=9, color=GREY)
    y += 14
    txt(p, MARGIN + 8, y, "Universite Paris 8  |  Mention Assez Bien", size=10, color=DARK)
    y += 20

    txt(p, MARGIN, y, "Master 1 -- Computer Vision & Machine Learning", size=11, font="hebo")
    txt(p, W - MARGIN - 80, y, "2022 - 2023", size=9, color=GREY)
    y += 14
    txt(p, MARGIN + 8, y, "Universite Paris 8  |  Mention Bien", size=10, color=DARK)
    y += 20

    txt(p, MARGIN, y, "Licence Informatique", size=11, font="hebo")
    txt(p, W - MARGIN - 80, y, "2019 - 2022", size=9, color=GREY)
    y += 14
    txt(p, MARGIN + 8, y, "Universite Paris 8  |  Mention Assez Bien", size=10, color=DARK)

    # -- Page 2 ----------------------------------------------------------------
    p2 = doc.new_page(width=W, height=H)
    fill_rect(p2, 0, 0, W, 45, color=BLUE)
    p2.insert_text(fitz.Point(MARGIN, 30), "NATSUKI Subaru -- CV (suite)",
                   fontname="hebo", fontsize=12, color=(1, 1, 1))

    y2 = 60

    # Competences techniques
    y2 = section(p2, y2, "Competences techniques")

    skills = [
        ("Langages",         "Python  |  SQL  |  Bash"),
        ("ML / DL",          "PyTorch  |  scikit-learn  |  HuggingFace Transformers"),
        ("NLP / RAG",        "LangChain  |  LangGraph  |  ChromaDB  |  sentence-transformers"),
        ("Computer Vision",  "OpenCV  |  YOLOv8  |  Detectron2"),
        ("MLOps",            "Docker  |  FastAPI  |  MLflow  |  GitHub Actions  |  GCP"),
        ("Evaluation LLM",   "RAGAS  |  TruLens"),
        ("Methode speciale", "Optimisation par essais-erreurs repetes (n = tres grand)"),
    ]
    for label, value in skills:
        fill_rect(p2, MARGIN, y2 + 2, MARGIN + 130, y2 + 14, color=LIGHT)
        txt(p2, MARGIN + 3, y2 + 1, label, size=9, font="hebo", color=BLUE)
        txt(p2, MARGIN + 138, y2 + 1, value, size=10, color=DARK)
        y2 += 18
    y2 += 8

    # Langues
    y2 = section(p2, y2, "Langues")
    for lang, level in [
        ("Japonais",  "Langue maternelle"),
        ("Francais",  "Courant (C1)"),
        ("Ancien elfe", "Bases conversationnelles"),
    ]:
        txt(p2, MARGIN + 3, y2, lang + " :", size=10, font="hebo", color=DARK)
        txt(p2, MARGIN + 100, y2, level, size=10, color=DARK)
        y2 += 16
    y2 += 8

    # Projets
    y2 = section(p2, y2, "Projets personnels (GitHub)")
    for name, desc in [
        ("personal-knowledge-assistant",
         "Agent RAG (LangGraph + ChromaDB + Groq) -- Q&A sur documents personnels."),
        ("nlp-paper-implementations",
         "Reproduction de TopicBERT et Molweni pour modelisation thematique et dialogue."),
        ("jutsu-detector",
         "Reconnaissance de gestes avec YOLOv8 + MediaPipe."),
    ]:
        txt(p2, MARGIN + 3, y2, "> " + name, size=10, font="hebo", color=BLUE)
        y2 += 14
        txt(p2, MARGIN + 12, y2, desc, size=10, color=DARK)
        y2 += 18

    # Divers
    y2 = section(p2, y2, "Informations complementaires")
    for line in [
        "Date de naissance : 01 avril 2000  |  Nationalite : Japonaise",
        "Permis B  |  Disponible immediatement",
        "Loisirs : jeux video, anime, entraitement de modeles en local la nuit",
    ]:
        txt(p2, MARGIN + 3, y2, line, size=10, color=DARK)
        y2 += 15

    out = OUT_DIR / "cv_subaru_natsuki.pdf"
    doc.save(str(out))
    doc.close()
    print("[OK] " + out.name)


# -- Document 2 -- Releve de notes M1 -----------------------------------------

def make_releve():
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)

    fill_rect(p, 0, 0, W, 100, color=(0.12, 0.20, 0.45))
    p.insert_text(fitz.Point(MARGIN, 35),
                  "UNIVERSITE PARIS 8 -- VINCENNES-SAINT-DENIS",
                  fontname="hebo", fontsize=13, color=(1, 1, 1))
    p.insert_text(fitz.Point(MARGIN, 55),
                  "UFR LLDI -- Departement Informatique",
                  fontname="helv", fontsize=11, color=(0.80, 0.88, 1.0))
    p.insert_text(fitz.Point(MARGIN, 75),
                  "Master 1 -- Computer Vision & Machine Learning  |  Annee 2022-2023",
                  fontname="helv", fontsize=10, color=(0.80, 0.88, 1.0))

    y = 118

    # Etudiant info box
    fill_rect(p, MARGIN, y, W - MARGIN, y + 52, color=LIGHT)
    txt(p, MARGIN + 10, y + 5,  "Nom :          NATSUKI",        size=10, font="hebo")
    txt(p, MARGIN + 10, y + 20, "Prenom :       Subaru",          size=10, color=DARK)
    txt(p, MARGIN + 10, y + 35, "N etudiant :   20221337",        size=10, color=DARK)
    txt(p, W // 2 + 10, y + 5,  "Formation :    M1 CV & ML",      size=10, font="hebo")
    txt(p, W // 2 + 10, y + 20, "Session :      Juin 2023",       size=10, color=DARK)
    txt(p, W // 2 + 10, y + 35, "Resultat :     ADMIS",           size=10, color=(0.0, 0.5, 0.1))
    y += 65

    # Table header
    col = [MARGIN, 290, 380, 450, 520]
    row_h = 18
    fill_rect(p, MARGIN, y, W - MARGIN, y + row_h, color=(0.10, 0.27, 0.60))
    for i, h in enumerate(["Unite d'enseignement", "Coeff.", "Note /20", "ECTS", "Resultat"]):
        p.insert_text(fitz.Point(col[i] + 4, y + 13), h,
                      fontname="hebo", fontsize=9, color=(1, 1, 1))
    y += row_h

    # Table rows -- Subaru est bon en RL (essais/erreurs) mais moyen ailleurs
    rows = [
        ("Fondamentaux de la Computer Vision",   "3", "12", "6",  "Valide"),
        ("Deep Learning & Reseaux de Neurones",  "3", "13", "6",  "Valide"),
        ("Traitement et Analyse d'Images",        "2", "11", "4",  "Valide"),
        ("Traitement Automatique du Langage",     "2", "10", "4",  "Valide"),
        ("Apprentissage par Renforcement",        "2", "19", "4",  "Valide"),
        ("Bases de donnees avancees",             "2", "10", "4",  "Valide"),
        ("Projet tutore de recherche",            "3", "14", "6",  "Valide"),
        ("Anglais scientifique",                  "1", "12", "2",  "Valide"),
        ("Stage de recherche (6 semaines)",       "4", "15", "4",  "Valide"),
    ]
    for idx, (ue, coeff, note, ects, res) in enumerate(rows):
        bg = LIGHT if idx % 2 == 0 else (1, 1, 1)
        fill_rect(p, MARGIN, y, W - MARGIN, y + row_h, color=bg)
        vals   = [ue, coeff, note, ects, res]
        sizes  = [9,  9,     10,   9,    9]
        fonts  = ["helv", "helv", "hebo", "helv", "helv"]
        rc     = (0.0, 0.5, 0.1) if res == "Valide" else (0.8, 0.1, 0.1)
        colors = [DARK, DARK, BLACK, DARK, rc]
        for i, (v, s, f, c) in enumerate(zip(vals, sizes, fonts, colors)):
            p.insert_text(fitz.Point(col[i] + 4, y + 13), v,
                          fontname=f, fontsize=s, color=c)
        y += row_h

    hline(p, y + 4)
    y += 14

    fill_rect(p, MARGIN, y, W - MARGIN, y + row_h, color=(0.88, 0.92, 0.98))
    txt(p, MARGIN + 4, y + 4, "TOTAL CREDITS VALIDES", size=10, font="hebo", color=BLUE)
    txt(p, col[3] + 4, y + 4, "40 / 40", size=10, font="hebo", color=BLUE)
    y += row_h + 8

    fill_rect(p, MARGIN, y, W - MARGIN, y + 28, color=LIGHT)
    txt(p, MARGIN + 10, y + 4,  "Moyenne generale ponderee :   12.94 / 20", size=11, font="hebo", color=DARK)
    txt(p, MARGIN + 10, y + 18, "Mention :   ASSEZ BIEN", size=11, font="hebo", color=(0.55, 0.35, 0.0))
    y += 40

    # Note du jury
    fill_rect(p, MARGIN, y, W - MARGIN, y + 22, color=(1.0, 0.97, 0.88))
    txt(p, MARGIN + 8, y + 5,
        "Note du jury : excellente progression en Apprentissage par Renforcement (19/20).",
        size=9, color=(0.5, 0.3, 0.0))

    y = H - 130
    hline(p, y, width=0.4, color=GREY)
    y += 10
    txt(p, MARGIN, y,
        "Ce releve de notes est delivre par l'Universite Paris 8, certifie conforme aux resultats officiels.",
        size=8, color=GREY)
    y += 14
    txt(p, MARGIN, y,
        "Document a usage administratif -- reproduit a titre d'exemple anonymise.",
        size=8, color=GREY)
    txt(p, W - 200, H - 80, "Le Directeur des etudes", size=9, color=DARK)
    txt(p, W - 200, H - 65, "Prof. M. Bernard",         size=9, color=DARK)
    txt(p, W - 200, H - 50, "Paris, le 15 juillet 2023", size=9, color=GREY)

    out = OUT_DIR / "releve_notes_m1.pdf"
    doc.save(str(out))
    doc.close()
    print("[OK] " + out.name)


# -- Document 3 -- Titre de sejour --------------------------------------------

def make_titre_sejour():
    doc = fitz.open()
    p = doc.new_page(width=W, height=H)

    fill_rect(p, 0, 0, W, H, color=(0.97, 0.97, 0.97))
    fill_rect(p, 0, 0, W, 80, color=(0.05, 0.18, 0.45))
    p.insert_text(fitz.Point(MARGIN, 32),
                  "REPUBLIQUE FRANCAISE",
                  fontname="hebo", fontsize=15, color=(1, 1, 1))
    p.insert_text(fitz.Point(MARGIN, 54),
                  "MINISTERE DE L'INTERIEUR",
                  fontname="helv", fontsize=11, color=(0.80, 0.90, 1.0))
    p.insert_text(fitz.Point(MARGIN, 70),
                  "Direction de l'immigration",
                  fontname="helv", fontsize=9, color=(0.70, 0.80, 1.0))

    y = 100
    p.insert_text(fitz.Point(MARGIN, y + 20),
                  "TITRE DE SEJOUR -- CARTE DE RESIDENT",
                  fontname="hebo", fontsize=16, color=(0.05, 0.18, 0.45))
    hline(p, y + 30, color=(0.05, 0.18, 0.45), width=1.2)
    y += 45

    fill_rect(p, MARGIN, y, W - MARGIN, y + 30, color=(0.05, 0.18, 0.45))
    txt(p, MARGIN + 10, y + 8, "N DE TITRE :  75-2024-003700-PREF-IDF",
        size=11, font="hebo", color=(1, 1, 1))
    y += 42

    def field_row(page, y_pos, label, value, highlight=False):
        bg = (0.88, 0.94, 1.0) if highlight else (1, 1, 1)
        fill_rect(page, MARGIN, y_pos, W - MARGIN, y_pos + 28, color=bg)
        txt(page, MARGIN + 8, y_pos + 6,  label, size=9,  font="hebo", color=GREY)
        txt(page, MARGIN + 8, y_pos + 17, value, size=11, font="hebo", color=BLACK)
        return y_pos + 30

    y = field_row(p, y, "NOM DE FAMILLE",       "NATSUKI",                       highlight=True)
    y = field_row(p, y, "PRENOM(S)",             "Subaru",                        highlight=False)
    y = field_row(p, y, "DATE DE NAISSANCE",    "01/04/2000",                    highlight=True)
    y = field_row(p, y, "LIEU DE NAISSANCE",    "Kawagoe, Japon",                highlight=False)
    y = field_row(p, y, "NATIONALITE",          "Japonaise",                     highlight=True)
    y = field_row(p, y, "SEXE",                 "M",                             highlight=False)
    y = field_row(p, y, "ADRESSE",
                  "4 rue de la Seconde Chance, 75005 Paris, France",             highlight=True)
    y += 8

    fill_rect(p, MARGIN, y, W - MARGIN, y + 28, color=(0.05, 0.18, 0.45))
    txt(p, MARGIN + 8, y + 6,  "TYPE DE TITRE",  size=9, font="hebo", color=(0.70, 0.82, 1.0))
    txt(p, MARGIN + 8, y + 17, "Etudiant -- Activite de formation superieure",
        size=11, font="hebo", color=(1, 1, 1))
    y += 30

    y = field_row(p, y, "DATE DE DELIVRANCE",   "15/09/2024",                   highlight=False)

    # Expiry -- highlighted
    fill_rect(p, MARGIN, y, W - MARGIN, y + 28, color=(1.0, 0.92, 0.92))
    txt(p, MARGIN + 8, y + 6,  "DATE D'EXPIRATION", size=9, font="hebo", color=(0.7, 0.1, 0.1))
    txt(p, MARGIN + 8, y + 17, "31/08/2027",
        size=13, font="hebo", color=(0.7, 0.1, 0.1))
    y += 32

    y = field_row(p, y, "DUREE DE VALIDITE",    "3 ans",                         highlight=False)
    y = field_row(p, y, "AUTORISATION DE TRAVAIL", "OUI -- 60h/semaine maximum", highlight=True)
    y += 10

    hline(p, y, color=GREY, width=0.4)
    y += 10
    fill_rect(p, MARGIN, y, W - MARGIN, y + 50, color=LIGHT)
    box(p, MARGIN + 8, y + 4, W - MARGIN - 8, y + 50,
        ("Ce titre de sejour a ete delivre conformement aux articles L.313-1 et suivants du "
         "Code de l'Entree et du Sejour des Etrangers et du Droit d'Asile (CESEDA). "
         "Toute falsification est passible de poursuites penales."),
        size=8, color=GREY)
    y += 58
    txt(p, MARGIN, y,
        "Document reproduit a titre d'exemple anonymise -- usage pedagogique uniquement.",
        size=8, color=GREY)

    out = OUT_DIR / "titre_sejour_exemple.pdf"
    doc.save(str(out))
    doc.close()
    print("[OK] " + out.name)


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating fake example documents in " + str(OUT_DIR) + " ...")
    make_cv()
    make_releve()
    make_titre_sejour()
    print("Done.")
