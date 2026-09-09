import warnings
warnings.filterwarnings("ignore")
import sys
import json
import os
import base64
import requests
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect
from PyQt5 import QtCore
from PyQt5.QtCore import QPoint, QRect
import vosk
import json
import queue
# 导入认证界面
from auth_ui import AuthDialog

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# ----------------------------- 自定义流式布局 -----------------------------
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.itemList = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Horizontal)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().left(), 2 * self.contentsMargins().top())
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spacing = self.spacing()
        for item in self.itemList:
            wid = item.widget()
            spaceX = spacing
            spaceY = spacing
            if lineHeight == 0:
                spaceY = 0
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0
            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())
        return y + lineHeight - rect.y()

# ----------------------------- 语音线程 -----------------------------


class SpeechThread(QThread):
    recognized = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        model_path = "vosk-model-small-cn-0.22"  # 模型路径
        if not os.path.exists(model_path):
            self.error.emit("未找到 Vosk 模型，请下载并解压")
            return

        model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(model, 16000)

        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=4000)
        stream.start_stream()

        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self.recognized.emit(text)
                        break
        except Exception as e:
            self.error.emit(str(e))
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
# ----------------------------- 检测线程 -----------------------------
class DetectThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, api_base, img_path, user_info, user_id):
        super().__init__()
        self.api_base = api_base
        self.img_path = img_path
        self.user_info = user_info
        self.user_id = user_id
    def run(self):
        try:
            url = f"{self.api_base}/api/skin_detect"
            with open(self.img_path, "rb") as f:
                files = {"file": f}
                data = {
                    "user_id": self.user_id,
                    "name": self.user_info["name"],
                    "age": str(self.user_info["age"]),
                    "gender": self.user_info["gender"]
                }
                resp = requests.post(url, files=files, data=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ----------------------------- 主窗口 -----------------------------
class SkinWindow(QMainWindow):
    def __init__(self, user_info, user_id):
        super().__init__()
        self.user_info = user_info
        self.user_id = user_id
        self.setWindowTitle("皮肤病智能检测系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setFont(QFont("Microsoft YaHei", 10))
        self.api_base = "http://127.0.0.1:8000"
        self.current_detect_result = None
        self.current_img_path = None
        self.knowledge2 = None

        # ---------- 美化后的全局样式表 ----------
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f4f8;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4f7df3, stop:1 #3a5fd9);
                color: white;
                border-radius: 12px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5f8dfa, stop:1 #4a6fe6);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a5fd9, stop:1 #2f4ec2);
            }
            QTextEdit, QLineEdit {
                background: white;
                border-radius: 12px;
                padding: 12px;
                border: 1px solid #dde3ed;
                font-size: 18px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 2px solid #4f7df3;
            }
            QTabWidget::pane {
                background: #f0f4f8;
                border: none;
                border-radius: 16px;
            }
            QTabBar::tab {
                background: #e4e9f2;
                color: #2c3e50;
                padding: 20px  45px;
                min-width: 150px;     /* 强制Tab最小宽度，避免被压缩 */
                min-height: 30px;     /* 强制Tab最小高度，解决垂直方向文字截断 */
                margin-right: 6px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: bold;
                font-size: 20px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #2563eb;
                border-bottom: 3px solid #2563eb;
            }
            QTableWidget {
                background: white;
                alternate-background-color: #f8fafd;
                border-radius: 12px;
                gridline-color: #e2e8f0;
                font-size: 16px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background: #eef2f7;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #cbd5e1;
                font-weight: bold;
                color: #1e293b;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QLabel {
                color: #1e293b;
                font-weight: 500;
            }
        """)

        self.load_knowledge2()

        self.tab = QTabWidget()
        self.setCentralWidget(self.tab)
        self.page1 = QWidget()
        self.page3 = QWidget()
        self.init_page1()
        self.init_page3()
        self.tab.addTab(self.page1, "📷 图片检测")
        self.tab.addTab(self.page3, "💬 智能问诊")

        self.load_history()

    def load_knowledge2(self):
        try:
            resp = requests.get(f"{self.api_base}/api/get_knowledge2", timeout=5)
            if resp.status_code == 200:
                self.knowledge2 = resp.json()
            else:
                self.knowledge2 = {"knowledge_base": []}
        except:
            self.knowledge2 = {"knowledge_base": []}

    def init_page1(self):
        main_layout = QHBoxLayout(self.page1)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ---------- 左侧面板 (图片 + 热力图 + 导出) ----------
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("""
            #leftPanel {
                background: white;
                border-radius: 20px;
                padding: 20px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        # 标题
        title_left = QLabel("📁 图片上传区")
        title_left.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b; padding-bottom: 5px;")
        left_layout.addWidget(title_left)

        self.btn_select = QPushButton("📂 选择图片")
        self.btn_select.setIconSize(QSize(24, 24))
        self.btn_select.clicked.connect(self.select_img)
        left_layout.addWidget(self.btn_select)

        self.img_label = QLabel("请上传图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(350, 250)
        self.img_label.setStyleSheet("""
            border: 2px dashed #cbd5e1;
            border-radius: 20px;
            background: #f8fafc;
            color: #64748b;
            font-size: 16px;
        """)
        left_layout.addWidget(self.img_label)

        # 热力图区域
        heat_title = QLabel("🔥 病灶热力图")
        heat_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b; margin-top: 10px;")
        left_layout.addWidget(heat_title)

        self.cam_label = QLabel("")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(350, 200)
        self.cam_label.setStyleSheet("""
            border: 2px dashed #cbd5e1;
            border-radius: 20px;
            background: #f8fafc;
            color: #64748b;
            font-size: 16px;
        """)
        left_layout.addWidget(self.cam_label)

        self.btn_export = QPushButton("📄 导出报告（HTML含图片）")
        self.btn_export.clicked.connect(self.export_report_html)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        left_layout.addStretch()

        # ---------- 右侧面板 (诊断结果 + 历史记录) ----------
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_panel.setStyleSheet("""
            #rightPanel {
                background: white;
                border-radius: 20px;
                padding: 20px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)

        # 诊断结果区域
        diag_title = QLabel("🔬 诊断结果")
        diag_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")
        right_layout.addWidget(diag_title)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(300)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background: #f9fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 15px;
                font-size: 18px;
            }
        """)
        right_layout.addWidget(self.result_text)

        # 历史记录区域
        history_header = QHBoxLayout()
        hist_title = QLabel("📋 历史记录")
        hist_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1e293b;")
        history_header.addWidget(hist_title)
        history_header.addStretch()
        self.btn_refresh_history = QPushButton("🔄 刷新")
        self.btn_refresh_history.clicked.connect(self.load_history)
        self.btn_refresh_history.setFixedWidth(120)  # 加宽以适应字体
        history_header.addWidget(self.btn_refresh_history)
        right_layout.addLayout(history_header)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["检测时间", "疾病", "置信度"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMinimumHeight(250)
        self.history_table.cellDoubleClicked.connect(self.show_detail_from_history)
        right_layout.addWidget(self.history_table)

        # 组装左右面板
        main_layout.addWidget(left_panel, 5)
        main_layout.addWidget(right_panel, 5)

    def select_img(self):
        path, _ = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.current_img_path = path
        pix = QPixmap(path).scaled(350, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)
        self.img_label.setStyleSheet("""
            border: 2px solid #4f7df3;
            border-radius: 20px;
            background: white;
        """)
        self.result_text.setText("诊断中，请稍候...")
        self.btn_export.setEnabled(False)
        self.cam_label.setText("生成热力图中...")
        self.cam_label.setStyleSheet("""
            border: 2px solid #fbbf24;
            border-radius: 20px;
            background: #fefce8;
            color: #92400e;
            font-weight: bold;
            font-size: 16px;
        """)

        self.detect_thread = DetectThread(self.api_base, path, self.user_info, self.user_id)
        self.detect_thread.finished.connect(self.on_detect_finished)
        self.detect_thread.error.connect(self.on_detect_error)
        self.detect_thread.start()

    def on_detect_finished(self, data):
        if "error" in data:
            self.result_text.setText(f"诊断失败：{data['error']}")
            self.cam_label.setText("热力图生成失败")
            self.cam_label.setStyleSheet("""
                border: 2px solid #ef4444;
                border-radius: 20px;
                background: #fef2f2;
                color: #991b1b;
                font-size: 16px;
            """)
            return

        top3 = data.get("top3", [])
        brief = data.get("brief_info", {})
        if not top3:
            self.result_text.setText("未获取到有效诊断结果")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        top1 = top3[0]
        text = f"检测时间：{now}\nTop1 疾病：{top1['disease_cn']}  置信度：{top1['score']*100:.1f}%\n\n"
        text += "【Top-3 预测】\n"
        for i, item in enumerate(top3, 1):
            text += f"{i}. {item['disease_cn']} ({item['disease_en']}) 置信度：{item['score']*100:.1f}%\n"
        text += "\n===== 简版疾病知识 =====\n"
        text += f"简介：{brief.get('简介', '无')}\n"
        text += f"症状：{brief.get('症状', '无')}\n"
        text += f"病因：{brief.get('病因', '无')}\n"
        text += f"用药：{brief.get('用药', '无')}\n"
        text += f"建议：{brief.get('建议', '无')}\n"
        self.result_text.setText(text)
        self.current_detect_result = data
        self.btn_export.setEnabled(True)

        cam_b64 = data.get("gradcam_base64", "")
        if cam_b64:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(base64.b64decode(cam_b64))).convert("RGB")
            qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(350, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.cam_label.setStyleSheet("""
                border: 2px solid #10b981;
                border-radius: 20px;
                background: white;
            """)
        else:
            self.cam_label.setText("未生成热力图")
            self.cam_label.setStyleSheet("""
                border: 2px solid #f59e0b;
                border-radius: 20px;
                background: #fffbeb;
                color: #92400e;
                font-size: 16px;
            """)

        self.load_history()

    def on_detect_error(self, error_msg):
        self.result_text.setText(f"诊断失败：{error_msg}\n请检查后端服务")
        self.cam_label.setText("病灶热力图")
        self.cam_label.setStyleSheet("""
            border: 2px solid #ef4444;
            border-radius: 20px;
            background: #fef2f2;
            color: #991b1b;
            font-size: 16px;
        """)

    def export_report_html(self):
        if not self.current_detect_result or not self.current_img_path:
            QMessageBox.warning(self, "提示", "无检测结果或图片路径")
            return
        detail_report = self.current_detect_result.get("detail_report", "")
        if not detail_report:
            QMessageBox.warning(self, "提示", "未生成详细报告")
            return

        with open(self.current_img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        ext = os.path.splitext(self.current_img_path)[1][1:].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

        gradcam_b64 = self.current_detect_result.get("gradcam_base64", "")
        heatmap_html = ""
        if gradcam_b64:
            heatmap_html = f"""
            <div style="margin-top: 20px;">
                <h3>病灶热力图</h3>
                <img src="data:image/png;base64,{gradcam_b64}" style="max-width: 100%; border-radius: 12px; border: 1px solid #ddd;">
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>皮肤病检测报告</title></head>
<body style="font-family: 'Microsoft YaHei', sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; background: #f9fafb;">
<div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: #2563eb;">皮肤病智能检测报告</h2>
    <div style="display: flex; gap: 24px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px;">
            <h3>检测图片</h3>
            <img src="data:{mime};base64,{img_b64}" style="max-width: 100%; border-radius: 12px; border: 1px solid #ddd;">
            {heatmap_html}
        </div>
        <div style="flex: 2;">
            <pre style="white-space: pre-wrap; font-family: inherit;">{detail_report}</pre>
        </div>
    </div>
</div>
</body>
</html>"""
        path, _ = QFileDialog.getSaveFileName(filter="HTML文件 (*.html)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_content)
            QMessageBox.information(self, "成功", "报告已保存为HTML文件，内含检测图片与热力图")

    def load_history(self):
        if not self.user_id:
            return
        try:
            resp = requests.get(f"{self.api_base}/api/history", params={"user_id": self.user_id}, timeout=10)
            resp.raise_for_status()
            records = resp.json().get("history", [])
            self.history_table.setRowCount(len(records))
            for i, rec in enumerate(records):
                self.history_table.setItem(i, 0, QTableWidgetItem(rec["time"]))
                self.history_table.setItem(i, 1, QTableWidgetItem(rec["disease"]))
                self.history_table.setItem(i, 2, QTableWidgetItem(rec["confidence"]))
                self.history_table.item(i, 0).setData(Qt.UserRole, rec["detail_report"])
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载历史失败：{str(e)}")

    def show_detail_from_history(self, row, col):
        detail = self.history_table.item(row, 0).data(Qt.UserRole)
        if not detail:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("详细报告")
        dlg.resize(700, 500)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText(detail)
        layout.addWidget(text_edit)
        dlg.setLayout(layout)
        dlg.exec_()

    def init_page3(self):
        layout = QVBoxLayout(self.page3)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

    # 标题
        title = QLabel("🤖 AI 智能问诊 · 递进式知识库")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #1e293b; padding: 10px 0;")
        layout.addWidget(title)

    # 快捷面板容器
        shortcut_card = QWidget()
        shortcut_card.setObjectName("shortcutCard")
        shortcut_card.setStyleSheet("""
            #shortcutCard {
                background: white;
                border-radius: 20px;
                padding: 15px;
            }
        """)
        card_layout = QVBoxLayout(shortcut_card)
        card_layout.setContentsMargins(10, 10, 10, 10)

    # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")
        self.shortcut_container = QWidget()
        self.shortcut_container.setStyleSheet("background: transparent;")
        self.flow_layout = FlowLayout(self.shortcut_container, margin=5, spacing=12)
        scroll_area.setWidget(self.shortcut_container)
        card_layout.addWidget(scroll_area)

    # 重置按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_reset = QPushButton("🏠 返回疾病列表")
        self.btn_reset.clicked.connect(self.reset_shortcut_panel)
        self.btn_reset.setFixedWidth(200)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                padding: 12px 22px;
            }
        """)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        layout.addWidget(shortcut_card)

    # 聊天记录区域
        chat_title = QLabel("💬 对话记录")
        chat_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b; margin-top: 10px;")
        layout.addWidget(chat_title)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background: white;
                border-radius: 22px;
                padding: 22px;
                font-size: 18px;
                border: 1px solid #e2e8f0;
            }
        """)
        layout.addWidget(self.chat_history)

    # ---------- 输入区域 ----------
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 30px;
                padding: 8px;
            }
        """)
        h = QHBoxLayout(input_container)
        h.setContentsMargins(15, 5, 10, 5)
        h.setSpacing(10)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("例如：痤疮怎么用药？湿疹日常护理建议...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                padding: 14px;
                font-size: 18px;
            }
        """)
        self.chat_input.returnPressed.connect(self.send_chat)

        self.btn_send = QPushButton("📤 发送")
        self.btn_send.clicked.connect(self.send_chat)
        self.btn_send.setFixedSize(100, 45)
        self.btn_send.setStyleSheet("""
            QPushButton {
                background: #2563eb;
                color: white;
                border-radius: 22px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:pressed {
                background: #1e40af;
            }
        """)

        self.btn_voice = QPushButton("🎤")
        self.btn_voice.setToolTip("语音输入")
        self.btn_voice.setFixedSize(50, 45)
        self.btn_voice.clicked.connect(self.voice_input)
        self.btn_voice.setStyleSheet("""
            QPushButton {
                background: #f3f4f6;
                color: #374151;
                border-radius: 22px;
                font-size: 20px;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
        """)

        h.addWidget(self.chat_input)
        h.addWidget(self.btn_send)
        h.addWidget(self.btn_voice)

        layout.addWidget(input_container)

        self.current_level = 0
        self.current_disease = None
        self.current_dimension = None
        self.disease_questions_map = {}

        self.parse_knowledge2()
        self.reset_shortcut_panel()
    def parse_knowledge2(self):
        if not self.knowledge2 or "knowledge_base" not in self.knowledge2:
            return
        for item in self.knowledge2["knowledge_base"]:
            disease_full = item.get("disease_name", "")
            cn_name = disease_full.split("—")[-1].split("/")[0].strip()
            if not cn_name:
                continue
            questions = item.get("questions", [])
            dim_list = []
            for q_group in questions:
                dim = q_group.get("dimension", "")
                subs = q_group.get("sub_questions", [])
                if dim and subs:
                    dim_list.append({"dimension": dim, "sub_questions": subs})
            if dim_list:
                self.disease_questions_map[cn_name] = dim_list

    def reset_shortcut_panel(self):
        self.current_level = 0
        self.current_disease = None
        self.current_dimension = None
        self.show_disease_list()

    def clear_flow_layout(self):
        while self.flow_layout.count():
            item = self.flow_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_disease_list(self):
        self.clear_flow_layout()
        diseases = list(self.disease_questions_map.keys())
        if not diseases:
            lbl = QLabel("未加载到知识库，请检查 knowledge2.json")
            lbl.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 18px;")
            self.flow_layout.addWidget(lbl)
            return
        for d in diseases:
            btn = QPushButton(d)
            btn.setStyleSheet("""
                QPushButton {
                    background: #eef2ff;
                    color: #1e3a8a;
                    border-radius: 20px;
                    padding: 14px 28px;
                    font-weight: 600;
                    font-size: 18px;
                    border: 1px solid #c7d2fe;
                }
                QPushButton:hover {
                    background: #dbeafe;
                    border-color: #a5b4fc;
                }
            """)
            btn.clicked.connect(lambda _, name=d: self.on_disease_selected(name))
            self.flow_layout.addWidget(btn)

    def on_disease_selected(self, disease_name):
        self.current_level = 1
        self.current_disease = disease_name
        self.show_dimension_list()

    def show_dimension_list(self):
        self.clear_flow_layout()
        dims = self.disease_questions_map.get(self.current_disease, [])
        for dim_item in dims:
            dim = dim_item["dimension"]
            btn = QPushButton(f"📌 {dim}")
            btn.setStyleSheet("""
                QPushButton {
                    background: #f0fdf4;
                    color: #166534;
                    border-radius: 20px;
                    padding: 14px 28px;
                    font-weight: 600;
                    font-size: 18px;
                    border: 1px solid #bbf7d0;
                }
                QPushButton:hover {
                    background: #dcfce7;
                    border-color: #86efac;
                }
            """)
            btn.clicked.connect(lambda _, d=dim: self.on_dimension_selected(d))
            self.flow_layout.addWidget(btn)

    def on_dimension_selected(self, dimension):
        self.current_level = 2
        self.current_dimension = dimension
        self.show_question_list()

    def show_question_list(self):
        self.clear_flow_layout()
        dims = self.disease_questions_map.get(self.current_disease, [])
        target = None
        for dim_item in dims:
            if dim_item["dimension"] == self.current_dimension:
                target = dim_item
                break
        if not target:
            return
        for q in target["sub_questions"]:
            short_text = q['question'][:30] + "…" if len(q['question']) > 30 else q['question']
            btn = QPushButton(f"❓ {short_text}")
            btn.setToolTip(q['question'])
            btn.setStyleSheet("""
                QPushButton {
                    background: #fff7ed;
                    color: #9a3412;
                    border-radius: 20px;
                    padding: 14px 28px;
                    font-weight: 500;
                    font-size: 18px;
                    border: 1px solid #fed7aa;
                }
                QPushButton:hover {
                    background: #ffedd5;
                    border-color: #fdba74;
                }
            """)
            btn.clicked.connect(lambda _, text=q['question']: self.send_question(text))
            self.flow_layout.addWidget(btn)

    def send_question(self, text):
        self.chat_input.setText(text)
        self.send_chat()

    def send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        # 统一字体大小为 18px
        self.chat_history.append(
            f"<b style='color:#2563eb; font-size:18px;'>🧑 我：</b>"
            f"<span style='font-size:18px;'>{text}</span>"
        )
        self.chat_input.clear()
        try:
            resp = requests.get(f"{self.api_base}/api/chat", params={"q": text}, timeout=10)
            resp.raise_for_status()
            answer = resp.json().get("answer", "暂无回答")
            self.chat_history.append(
                f"<b style='color:#059669; font-size:18px;'>🤖 AI：</b>"
                f"<span style='font-size:18px;'>{answer}</span><br>"
            )
        except Exception as e:
            self.chat_history.append(
                f"<b style='color:#dc2626; font-size:18px;'>⚠️ AI：</b>"
                f"<span style='font-size:18px;'>连接失败，请检查后端服务 ({str(e)})</span><br>"
            )

    def voice_input(self):
        if not SPEECH_AVAILABLE:
            QMessageBox.warning(self, "提示", "未安装SpeechRecognition库，无法使用语音输入。\n请执行：pip install SpeechRecognition pyaudio")
            return
        self.voice_thread = SpeechThread()
        self.voice_thread.recognized.connect(self.on_voice_recognized)
        self.voice_thread.error.connect(lambda e: QMessageBox.warning(self, "语音识别错误", e))
        self.voice_thread.start()

    def on_voice_recognized(self, text):
        self.chat_input.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    auth_dlg = AuthDialog()
    if auth_dlg.exec_() == QDialog.Accepted:
        user_info = auth_dlg.user_info
        user_id = user_info["user_id"]
        w = SkinWindow(user_info, user_id)
        w.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)