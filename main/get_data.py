from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time
from tqdm import tqdm

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)

TOTAL_QUESTIONS = 305

base_url = "https://pass-dmv-test.com/quiz-{}-zh"

questions = []

output_file = "./data/questions.json"


def get_question_data():
    question_data = {}

    question_element = driver.find_element(By.CLASS_NAME, "card-title")
    question_data["question"] = question_element.text.strip()

    image_url = ""
    try:
        image_element = driver.find_element(By.CSS_SELECTOR, "figure.figure img")
        image_url = image_element.get_attribute("src")
    except:
        pass
    question_data["image_url"] = image_url

    options = []
    correct_answer = ""
    options_elements = driver.find_elements(By.CSS_SELECTOR, ".list-group-item")

    option_labels = ['A.', 'B.', 'C.', 'D.']

    for index, option_element in enumerate(options_elements):
        label = option_element.find_element(By.TAG_NAME, "label")
        option_text = f"{option_labels[index]} {label.text.strip()}"
        options.append(option_text)

        if option_element.find_elements(By.CSS_SELECTOR, "span#js-checker-icon") or option_element.find_elements(
                By.CSS_SELECTOR, "input[value='1']"):
            correct_answer = option_text

    question_data["options"] = options
    question_data["correct_answer"] = correct_answer.split('.')[0]

    return question_data


for i in tqdm(range(1, TOTAL_QUESTIONS + 1), desc="get process", unit="question"):
    try:
        driver.get(base_url.format(i))
        time.sleep(2)
        question_data = get_question_data()
        questions.append(question_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"\nnum {i} question error: {e}")
        break

driver.quit()

print(f"\nsave to {output_file}")
