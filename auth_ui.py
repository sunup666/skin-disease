# auth_ui.py
import sys
import requests
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
    QMessageBox, QTabWidget, QWidget, QFormLayout, QApplication
)
from PyQt5.QtCore import Qt

API_BASE = "http://127.0.0.1:8000"


class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户注册")
        self.setFixedSize(350, 280)
        self.setModal(True)
        layout = QVBoxLayout()

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("真实姓名")
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("年龄")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["男", "女"])
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("设置密码")
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit.setPlaceholderText("确认密码")

        form.addRow("姓名", self.name_edit)
        form.addRow("年龄", self.age_edit)
        form.addRow("性别", self.gender_combo)
        form.addRow("密码", self.password_edit)
        form.addRow("确认密码", self.confirm_edit)

        layout.addLayout(form)

        self.btn_register = QPushButton("注册")
        self.btn_register.clicked.connect(self.do_register)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def do_register(self):
        name = self.name_edit.text().strip()
        age_text = self.age_edit.text().strip()
        gender = self.gender_combo.currentText()
        pwd = self.password_edit.text()
        confirm = self.confirm_edit.text()

        if not name or not age_text or not pwd:
            QMessageBox.warning(self, "提示", "请完整填写所有信息")
            return
        if pwd != confirm:
            QMessageBox.warning(self, "提示", "两次输入的密码不一致")
            return
        try:
            age = int(age_text)
        except ValueError:
            QMessageBox.warning(self, "提示", "年龄请输入数字")
            return

        try:
            resp = requests.post(f"{API_BASE}/api/register", data={
                "name": name,
                "age": age,
                "gender": gender,
                "password": pwd
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            QMessageBox.information(
                self, "注册成功",
                f"注册成功！\n您的用户编号：{data['user_id']}\n请妥善保管，用于登录。"
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "注册失败", f"注册失败：{str(e)}")


from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
                             QPushButton, QMessageBox)
import requests



class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户登录")
        self.setFixedSize(350, 280)  # 加高窗口适配新增输入框
        self.setModal(True)
        layout = QVBoxLayout()

        form = QFormLayout()
        self.user_id_edit = QLineEdit()
        self.user_id_edit.setPlaceholderText("用户编号")
        
        # 新增：姓名输入框
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("姓名")
        
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("密码")

        # 添加到表单布局
        form.addRow("用户编号", self.user_id_edit)
        form.addRow("姓名", self.name_edit)  # 新增行
        form.addRow("密码", self.password_edit)

        layout.addLayout(form)

        self.btn_login = QPushButton("登录")
        self.btn_login.clicked.connect(self.do_login)
        layout.addWidget(self.btn_login)

        self.setLayout(layout)
        self.user_info = None

    def do_login(self):
        # 获取三个输入框内容
        user_id = self.user_id_edit.text().strip()
        name = self.name_edit.text().strip()  # 新增
        pwd = self.password_edit.text()
        
        # 校验：三项都不能为空
        if not user_id or not name or not pwd:
            QMessageBox.warning(self, "提示", "请输入完整的用户编号、姓名和密码")
            return

        try:
            # 请求接口时带上姓名参数
            resp = requests.post(f"{API_BASE}/api/login", data={
                "user_id": user_id,
                "name": name,  # 新增
                "password": pwd
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.user_info = {
                "user_id": data["user_id"],
                "name": data["name"],
                "age": data["age"],
                "gender": data["gender"]
            }
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "登录失败", f"登录失败：{str(e)}")


class AuthDialog(QDialog):
    """包含注册和登录选项卡的对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("皮肤病智能检测系统 - 认证")
        self.setFixedSize(400, 350)
        self.setModal(True)

        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.register_tab = RegisterDialog()
        self.login_tab = LoginDialog()
        self.tabs.addTab(self.register_tab, "注册")
        self.tabs.addTab(self.login_tab, "登录")
        layout.addWidget(self.tabs)

        # 注册成功或登录成功后关闭对话框
        self.register_tab.accepted.connect(self.on_auth_success)
        self.login_tab.accepted.connect(self.on_auth_success)

        self.setLayout(layout)
        self.user_info = None

    def on_auth_success(self):
        if self.tabs.currentWidget() == self.register_tab:
            # 注册成功后自动填充用户编号到登录页，提示用户登录
            QMessageBox.information(self, "提示", "注册成功，请切换到登录页使用新编号登录。")
            self.tabs.setCurrentIndex(1)
        else:
            self.user_info = self.login_tab.user_info
            self.accept()


if __name__ == "__main__":
    # 测试用
    app = QApplication(sys.argv)
    dlg = AuthDialog()
    if dlg.exec_() == QDialog.Accepted:
        print("认证成功:", dlg.user_info)
    else:
        print("取消")