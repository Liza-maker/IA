from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from reportlab.lib import colors
from reportlab.lib.units import cm
import os

pdfmetrics.registerFont(TTFont('Anton', 'static/Anton-Regular.ttf'))

class HeaderWithBackground(Flowable):
    """
    Logo sur fond gris à gauche et rectangle rouge avec le titre à droite.
    """
    def __init__(self, logo_path=None, title_text="COMMUNIQUÉ DE PRESSE"):
        Flowable.__init__(self)
        self.logo_path = logo_path
        self.title_text = title_text
        self.height = 3.5 * cm
        self.page_width = A4[0]

    def draw(self):
        c = self.canv
        x = 0
        y = 0

        logo_size = 3 * cm
        offset_x = 0

        # Fond gris pour le logo
        logo_bg_width = logo_size + 1 * cm
        c.setFillColor(colors.HexColor("#f2f2f2"))
        c.rect(x, y, logo_bg_width, self.height, stroke=0, fill=1)

        # Logo
        if self.logo_path and os.path.exists(self.logo_path):
            c.drawImage(
                self.logo_path,
                x + 0.5 * cm, y + (self.height - logo_size) / 2,
                width=logo_size, height=logo_size,
                preserveAspectRatio=True, mask='auto'
            )
        offset_x = logo_bg_width

        # Rectangle rouge à droite
        rect_width = self.page_width - offset_x
        c.setFillColor(colors.HexColor("#ec2423"))
        c.rect(offset_x, y, rect_width, self.height, stroke=0, fill=1)

        # Texte en blanc
        c.setFillColor(colors.white)
        c.setFont("Anton", 35)
        text_x = offset_x + 1 * cm
        text_y = y + self.height / 2 - 10
        c.drawString(text_x, text_y, self.title_text)

class FooterGraphics(Flowable):
    """
    Petites formes graphiques inclinées en bas à droite.
    """
    def __init__(self):
        Flowable.__init__(self)
        self.page_width, self.page_height = A4

    def draw(self):
        c = self.canv
        x_right = self.page_width
        y_bottom = 0

        # Triangle bleu foncé
        c.setFillColor(colors.HexColor("#1d0e77"))
        c.saveState()
        c.translate(x_right - 5*cm, y_bottom)
        c.rotate(-45)
        c.rect(0, 0, 2*cm, 8*cm, stroke=0, fill=1)
        c.restoreState()

        # Petit losange rouge
        c.setFillColor(colors.HexColor("#ec2423"))
        c.saveState()
        c.translate(x_right - 3*cm, y_bottom + 2*cm)
        c.rotate(-45)
        c.rect(0, 0, 1*cm, 3*cm, stroke=0, fill=1)
        c.restoreState()

def generate_pdf(content, output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=0*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        spaceAfter=6
    )

    elements = []

    # En-tête avec logo et rectangle rouge
    logo_path = "static/images/logo_asbh.png"
    elements.append(HeaderWithBackground(logo_path))
    elements.append(Spacer(1, 12))

    # Paragraphes
    paragraphs = content.strip().split('\n\n')
    for para in paragraphs:
        para = para.strip().replace('\n', ' ')
        elements.append(Paragraph(para, normal_style))
        elements.append(Spacer(1, 6))

    # Ajout des formes graphiques en bas
    elements.append(Spacer(1, 100))  # Ajuste la hauteur pour éviter que le texte chevauche les graphiques
    elements.append(FooterGraphics())

    doc.build(elements)

if __name__ == "__main__":
    content = """
L'UBB HISTOIRE : LA TEAM FRANÇAISE REMPORTE SA PREMIÈRE COUPE D'EUROPE !

Le 24 mai 2025, à Cardiff, l'Union Bordeaux-Bègles a inscrit son nom dans les annales du rugby en remportant la Champions Cup face aux Anglais de Northampton. Avec deux nouveaux essais de Damian Penaud, les Girondins ont dominé le match (20-28) et ont obtenu leur premier titre continental.

Cette victoire est le fruit d'un travail collectif exceptionnel de l'équipe dirigée par Yannick Bru. Les joueurs bordelais ont montré une patience et une détermination extrêmes.

Nous encourageons tous les supporters français à célébrer cette victoire historique !
"""
    output_pdf = "communique_presse_test6.pdf"
    generate_pdf(content, output_pdf)
    print(f"PDF généré : {output_pdf}")
