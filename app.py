
import gradio as gr
import pickle
import numpy as np


# โหลดโมเดล
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# mapping ภาษาอังกฤษ → แปลไทย
job_translation = {
    'Actor / Actress': 'นักแสดง',
    'Actuary': 'นักคณิตศาสตร์ประกันภัย',
    'Anthropologist': 'นักมานุษยวิทยา',
    'Archeologist': 'นักโบราณคดี',
    'Artist': 'ศิลปิน',
    'Astronomer': 'นักดาราศาสตร์',
    'Athlete': 'นักกีฬา',
    'Audiologist': 'นักตรวจการได้ยิน',
    'Banking': 'พนักงานธนาคาร',
    'Broadcaster': 'ผู้ประกาศ',
    'Business Analyst': 'นักวิเคราะห์ธุรกิจ',
    'Business manager': 'ผู้จัดการธุรกิจ',
    'Chartered Accountant': 'นักบัญชีรับอนุญาต',
    'Chief financial officer': 'ประธานเจ้าหน้าที่ฝ่ายการเงิน',
    'Company secretary': 'เลขานุการบริษัท',
    'Computer analyst': 'นักวิเคราะห์ระบบคอมพิวเตอร์',
    'Computer programmer': 'โปรแกรมเมอร์',
    'Consultant': 'ที่ปรึกษา',
    'Counselor': 'นักให้คำปรึกษา',
    'Criminologist': 'นักอาชญาวิทยา',
    'Dancer': 'นักเต้น',
    'Database designer': 'นักออกแบบฐานข้อมูล',
    'Economist': 'นักเศรษฐศาสตร์',
    'Editor': 'บรรณาธิการ',
    'Engineer': 'วิศวกร',
    'Fashion Designer': 'นักออกแบบแฟชั่น',
    'Financial Advisor': 'ที่ปรึกษาทางการเงิน',
    'Geologist': 'นักธรณีวิทยา',
    'Graphic Designer': 'นักออกแบบกราฟิก',
    'Historian': 'นักประวัติศาสตร์',
    'Interior Decorator': 'มัณฑนากร',
    'Internal auditor': 'ผู้ตรวจสอบภายใน',
    'Journalist': 'นักข่าว',
    'Language Teacher': 'ครูสอนภาษา',
    'Lawyer': 'ทนายความ',
    'Leader': 'ผู้นำ',
    'Librarian': 'บรรณารักษ์',
    'Logistics manager': 'ผู้จัดการโลจิสติกส์',
    'Manager': 'ผู้จัดการ',
    'Marine Biologist': 'นักชีววิทยาทางทะเล',
    'Marketing': 'นักการตลาด',
    'Mathematician': 'นักคณิตศาสตร์',
    'Mechanic': 'ช่างซ่อมเครื่องกล',
    'Medical': 'บุคลากรทางการแพทย์',
    'Middle, Higher School Teacher and Professors': 'ครูมัธยม/อาจารย์',
    'Militry': 'ทหาร',
    'Music teacher': 'ครูสอนดนตรี',
    'Nature photographer': 'ช่างภาพธรรมชาติ',
    'Para Medical (physiotherapy, occupational theropy, audio and speech language theropy, nursing)': 'บุคลากรทางการแพทย์เสริม',
    'Para Militry (https://en.wikipedia.org/wiki/Paramilitary_forces_of_India)': 'กองกำลังกึ่งทหาร',
    'Pharmacist': 'เภสัชกร',
    'Philosopher': 'นักปรัชญา',
    'Physical Therapist': 'นักกายภาพบำบัด',
    'Physician': 'แพทย์',
    'Physicist': 'นักฟิสิกส์',
    'Pilot': 'นักบิน',
    'Poet': 'กวี',
    'Police Force (Spies, CBI officials, CID, Detectives)': 'เจ้าหน้าที่ตำรวจ/นักสืบ',
    'Politician': 'นักการเมือง',
    'Pre Primary Teacher (2020 NEP and Mental Health)': 'ครูอนุบาล',
    'Primary Teacher': 'ครูประถม',
    'Psychologist': 'นักจิตวิทยา',
    'Receptionist': 'พนักงานต้อนรับ',
    'Recording engineer': 'วิศวกรเสียง',
    'Research analyst': 'นักวิเคราะห์วิจัย',
    'Sales Representative': 'ตัวแทนฝ่ายขาย',
    'Social Worker': 'นักสังคมสงเคราะห์',
    'Sound editor': 'ผู้ตัดต่อเสียง',
    'Stock Broker': 'นายหน้าซื้อขายหุ้น',
    'Technician': 'ช่างเทคนิค',
    'Veterinarian': 'สัตวแพทย์',
    'Writer': 'นักเขียน'
}

# ฟังก์ชันทำนาย
def predict(p1, p2, p3, p4, p5, p6, p7, p8):
    X = np.array([[p1, p2, p3, p4, p5, p6, p7, p8]])
    label = model.predict(X)[0]
    job_en = le.inverse_transform([label])[0]
    job_th = job_translation.get(job_en.strip(), "ไม่พบคำแปล")
    return f"{job_en}\n({job_th})"

# อินพุตแบบปุ่ม + - และช่วง 0–20
inputs = [
    gr.Number(label="Linguistic", minimum=0, maximum=20, step=1),
    gr.Number(label="Musical", minimum=0, maximum=20, step=1),
    gr.Number(label="Bodily", minimum=0, maximum=20, step=1),
    gr.Number(label="Logical-Mathematical", minimum=0, maximum=20, step=1),
    gr.Number(label="Spatial-Visualization", minimum=0, maximum=20, step=1),
    gr.Number(label="Interpersonal", minimum=0, maximum=20, step=1),
    gr.Number(label="Intrapersonal", minimum=0, maximum=20, step=1),
    gr.Number(label="Naturalist", minimum=0, maximum=20, step=1),
]

# สร้าง Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="อาชีพที่เหมาะสม (EN / TH)"),
    title="อาชีพที่ใช่ ใครก็ชอบ",
    description="ใส่คะแนนแต่ละด้านเพื่อทำนายอาชีพที่เหมาะสม"
)

demo.launch()

