import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# ================= PDF IMPORT =================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# ================= KONFIGURASI =================
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HISTORY_FILE'] = 'history.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================= LOAD MODEL =================
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D

class FixedDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

model = load_model('model/mobilenetv2_model.h5', custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})

CLASS_NAMES = ['daun_kering', 'daun_menguning', 'layu_daun', 'ulat_api']


# ================= FUNGSI PREDIKSI =================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    confidence = round(float(np.max(pred)) * 100, 2)
    label = CLASS_NAMES[np.argmax(pred)]

    suggestions = {
        'daun_kering': 'Pastikan tanaman mendapat cukup air dan nutrisi.',
        'daun_menguning': 'Periksa kemungkinan kekurangan nitrogen atau penyakit awal.',
        'layu_daun': 'Gunakan fungisida alami dan buang daun yang terinfeksi.',
        'ulat_api': 'Gunakan pestisida alami seperti daun mimba atau lampu perangkap ulat.'
    }

    suggestion = suggestions.get(label, "Tanaman dalam kondisi baik.")
    return label, confidence, suggestion


# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "Tidak ada file dipilih."

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        label, confidence, suggestion = predict_image(filepath)
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history = []
        if os.path.exists(app.config['HISTORY_FILE']):
            try:
                with open(app.config['HISTORY_FILE'], 'r') as f:
                    history = json.load(f)
            except:
                history = []

        entry = {
            "filename": filename,
            "label": label,
            "confidence": confidence,
            "suggestion": suggestion,
            "time": time_now
        }

        history.append(entry)

        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f, indent=4)

        return render_template(
            'result.html',
            filename=filename,
            label=label,
            confidence=confidence,
            suggestion=suggestion,
            image_url=url_for('static', filename=f"uploads/{filename}")
        )

    return render_template('detect.html')


@app.route('/history')
def history_view():
    history = []
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r') as f:
                history = json.load(f)
        except:
            history = []

    return render_template('history.html', history=history)


@app.route('/delete_history')
def delete_history():
    if os.path.exists(app.config['HISTORY_FILE']):
        os.remove(app.config['HISTORY_FILE'])
    return redirect(url_for('history_view'))


# ================= EXPORT PDF FORMAL & RAPI =================
@app.route('/export_pdf/<filename>')
def export_pdf(filename):

    if not os.path.exists(app.config['HISTORY_FILE']):
        return "Tidak ada riwayat."

    with open(app.config['HISTORY_FILE'], 'r') as f:
        history = json.load(f)

    data = next((x for x in history if x["filename"] == filename), None)
    if not data:
        return "Data tidak ditemukan."

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_result.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # ===== JUDUL =====
    elements.append(Paragraph("<b>LAPORAN HASIL DETEKSI DAUN KELAPA</b>", styles["Title"]))
    elements.append(Spacer(1, 0.4 * inch))

    # ===== TABEL DATA RAPI =====
    normal_style = styles["Normal"]

    table_data = [
        [Paragraph("<b>Nama File</b>", normal_style), Paragraph(filename, normal_style)],
        [Paragraph("<b>Label Prediksi</b>", normal_style), Paragraph(data['label'], normal_style)],
        [Paragraph("<b>Tingkat Kepercayaan</b>", normal_style), Paragraph(f"{data['confidence']} %", normal_style)],
        [Paragraph("<b>Saran Penanganan</b>", normal_style), Paragraph(data['suggestion'], normal_style)],
        [Paragraph("<b>Waktu Deteksi</b>", normal_style), Paragraph(data['time'], normal_style)],
    ]

    table = Table(table_data, colWidths=[180, 330])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.6, colors.grey),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.6 * inch))

    # ===== GAMBAR =====
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(image_path):
        elements.append(Paragraph("<b>Gambar Daun yang Dianalisis:</b>", styles["Heading3"]))
        elements.append(Spacer(1, 0.3 * inch))

        img = Image(image_path)
        img.drawHeight = 3 * inch
        img.drawWidth = 4.5 * inch
        elements.append(img)

    elements.append(Spacer(1, 0.7 * inch))

    # ===== FOOTER =====
    elements.append(Paragraph(
        "Sistem Deteksi Hama dan Penyakit Daun Kelapa Berbasis Convolutional Neural Network (MobileNetV2) - KelapaCare",
        styles["Normal"]
    ))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)


# ================= MAIN =================
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
