from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import numpy as np
import cv2
import requests
import urllib
import uuid
from threading import Thread
from prediction import predict, prediction_model
def download(class_id_name):
	op = webdriver.ChromeOptions()
	# driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
	driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)
	driver.get('https://mydtu.duytan.edu.vn/Signin.aspx')
	# driver.find_element('txtUser').send_keys('maiduclong')
	txtuser = driver.find_element('id', 'txtUser').send_keys("")
	txtPass = driver.find_element('id', 'txtPass').send_keys("")
	time.sleep(10)
	driver.get('https://mydtu.duytan.edu.vn/sites/index.aspx?p=home_registeredall&semesterid=79&yearid=78')
	i = 0
	while True:
		img_capt = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "btn-responsive")))
		url = img_capt.get_attribute('src')


		s = requests.Session()
		# Set correct user agent
		selenium_user_agent = driver.execute_script("return navigator.userAgent;")
		s.headers.update({"user-agent": selenium_user_agent})

		for cookie in driver.get_cookies():
		    s.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

		response = s.get(url)
		image = np.asarray(bytearray(response.content), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray,(200, 50))
		capcha_text= predict(gray, prediction_model)
		txtMaDangKy = driver.find_element('id', 'ctl00_PlaceHolderContentArea_ctl00_ctl01_txt_ClassID').send_keys("Mailong2$")
		txtCapcha = driver.find_element('id', 'ctl00_PlaceHolderContentArea_ctl00_ctl01_txtCaptchar').send_keys(capcha_text)
		btnadd = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "btnadd")))
		btnadd.click()
		alert = WebDriverWait(driver, 30).until(EC.alert_is_present())
		alert.accept()
		while True:
			table_content = driver.find_element('class name', 'tab-content')
			div_ThongBao = table_content.find_element('id', 'displayThongBao')
			if div_ThongBao.text != '':
				print(div_ThongBao.text)
				break
		if 
		# driver.refresh()
		i += 1
		# print(i, end='\r')




class_id_name = ['MGT396202302013',
			'MKT402202302001',
			'MKT403202302006'
]
ts = []
for i in range(1):
    t = Thread(target=download)
    t.setDaemon = True
    ts.append(t)
for t in ts:
    t.start()


