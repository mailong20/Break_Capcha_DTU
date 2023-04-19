from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import numpy as np
import cv2
import requests
from threading import Thread
from prediction import predict, prediction_model
from datetime import datetime



class DTU_Tab:
	def __init__(self, class_name, luong):
		self.class_name = class_name
		self.luong = luong
		self.url_register = 'https://mydtu.duytan.edu.vn/sites/index.aspx?p=home_registeredall&semesterid=80&yearid=78'
		# self.url_register = 'https://mydtu.duytan.edu.vn/sites/index.aspx?p=home_registeredall&semesterid=79&yearid=78'
		op = webdriver.ChromeOptions()
		self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)
		while True:
			time_str = '00:00:00'
			time_str1 = '24:00:00'
			time_str2 = '12:00:00'
			now = datetime.now().strftime("%H:%M:%S")
			if time_str == str(now) or time_str1 == str(now) or time_str2 == str(now):
				break
		self.driver.get('https://mydtu.duytan.edu.vn/Signin.aspx')
		while True:
			img_capte = WebDriverWait(self.driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR,'[alt="Captcha"]')))
			if img_capte:
				if img_capte.get_attribute('src'):
					break
			self.driver.refresh()
		self.login_dtu()
		self.register_class()


	def login_dtu(self):
		txtuser = self.driver.find_element('id', 'txtUser').send_keys("")
		txtPass = self.driver.find_element('id', 'txtPass').send_keys("")
		txtCaptcha = self.driver.find_element('id', 'txtCaptcha')
		while True:
			txtCaptcha = self.driver.find_element('id', 'txtCaptcha')
			if len(txtCaptcha.get_attribute('value')) == 4:
					btn_login = self.driver.find_element(By.CLASS_NAME, 'button')
					print(btn_login)
					btn_login.click()
					break
		login_click = WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.ID, 'subheader')))


	def register_class(self):
		self.driver.get(self.url_register)
		while True:
			table_content = WebDriverWait(self.driver, 120).until(EC.presence_of_element_located((By.CLASS_NAME, 'tab-content')))
			if table_content:
				div_ThongBao = table_content.find_element('class name', 'errormes')

				if div_ThongBao.text == 'Bạn không được phép Đăng ký Lớp':
					time.sleep(2)
					self.driver.refresh()
				else:
					break
		for index, class_id in enumerate(self.class_name):
			while True:
				kq = self.register(class_id)
				if kq != 'MÃ XÁC NHẬN không chính xác. Vui lòng nhập lại!':
					print(kq)
					self.ouput_log(index, class_id, kq)
					break
		self.driver.close()			


	def register(self, class_id):
		
		img_capt = WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.ID, "imgCapt")))
		url = img_capt.get_attribute('src')
		s = requests.Session()
		# Set correct user agent
		selenium_user_agent = self.driver.execute_script("return navigator.userAgent;")
		s.headers.update({"user-agent": selenium_user_agent})

		for cookie in self.driver.get_cookies():
			s.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

		response = s.get(url)
		image = np.asarray(bytearray(response.content), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray,(200, 50))
		capcha_text= predict(gray, prediction_model)
		print('|',capcha_text,'|')
		if len(capcha_text) != 4:
			capcha_text = '1234'
		print(capcha_text)
		txtMaDangKy = self.driver.find_element('id', 'ctl00_PlaceHolderContentArea_ctl00_ctl01_txt_ClassID')
		txtMaDangKy.clear()
		txtMaDangKy.send_keys(class_id)

		txtCapcha = self.driver.find_element('id', 'ctl00_PlaceHolderContentArea_ctl00_ctl01_txtCaptchar')
		txtCapcha.clear()
		txtCapcha.send_keys(capcha_text)

		btnadd = WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.NAME, "btnadd")))
		btnadd.click()
		alert = WebDriverWait(self.driver, 60).until(EC.alert_is_present())
		msa = alert.text
		time.sleep(0.5)
		alert.accept()
		print('msa', msa)
		while True:
			time.sleep(0.5)
			table_content = self.driver.find_element('class name', 'tab-content')
			div_ThongBao = table_content.find_element('id', 'displayThongBao')
			if div_ThongBao.text != '':
				return div_ThongBao.text

	def ouput_log(self, index, class_id, kq):
		with open('log.txt', 'a', encoding='utf-8') as f:
			log = f'L: {self.luong} {str(index)} {class_id} {str(kq)}\n'
			f.write(log)



# class_id_name = ['MGT396202302013',
# 			'MKT402202302001',
# 			'MKT403202302006'
# ]
def main(class_id_name, luong):
	m = DTU_Tab(class_id_name, luong)
# class_id_names = [['11',
# 			'22',
# 			'33'
# ], ['22',
# 			'11',
# 			'33'
# ],
# ['33',
# 			'22',
# 			'11'
# ]]


class_id_names = [['MGT396202302013',
			'MKT402202302001',
			'MKT403202302006'
], ['MKT402202302001',
			'MGT396202302013',
			'MKT403202302006'
],
['MKT403202302006',
			'MKT402202302001',
			'MGT396202302013'
]]




ts = []
for i in range(1):
    t = Thread(target=main, args=(class_id_names[i], i))
    t.setDaemon = True
    ts.append(t)
for t in ts:
    t.start()