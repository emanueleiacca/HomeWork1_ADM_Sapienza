# Anaconda ADMenv to run it
#usr: emanueleiaccarino.ei@gmail.com
#passw: ###
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
from selenium.webdriver.chrome.options import Options
from PyPDF2 import PdfMerger
import getpass
import pyperclip
from selenium.webdriver.common.keys import Keys
import glob



start_time = time.time()
print("*****************************************************************************************")
print("*****************************************************************************************")
print("*************                                                               *************")
print("*************       Welcome to Pdf Extractor for HackerRank Submissions     *************")
print("*************                    Tool created by                            *************")
print("*************                https://github.com/vokbuda                     *************")
print("*************                                                               *************")
print("*****************************************************************************************")
print("*****************************************************************************************")
import os
import glob

# Get the current working directory
current_directory = os.getcwd()

# Find all files in the current directory starting with 'Submissions_Hackerrank'
files_to_remove = glob.glob(os.path.join(current_directory, 'Submissions_Hackerrank*'))
if os.path.exists('merged_output.pdf'):
    os.remove('merged_output.pdf')
if os.path.exists('scripts.py'):
    os.remove('scripts.py')

# Loop through and remove each file
for file in files_to_remove:
    try:
        os.remove(file)
        print(f"Removed: {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"Error occurred while removing {file}: {e}")


while True:
    try:
        number_of_pages_hacker_rank = int(input("Please insert number of pages you want to extract: \n"))
        if number_of_pages_hacker_rank <= 0:
            raise ValueError("The number of pages must be a positive integer.")
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")

username = input("Enter your username(hackerrank): \n")
password = getpass.getpass("Insert your password(hackerrank): \n")

current_directory = os.getcwd()
# Use ChromeDriverManager to handle versioning automatically
chrome_options = Options()
chrome_options.add_argument('--kiosk-printing')  # Enable silent printing
chrome_options.add_argument("--log-level=3")  # Set log level to severe
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

prefs = {
    "printing.print_preview_sticky_settings.appState": '{"recentDestinations":[{"id":"Save as PDF","origin":"local","account":""}],"selectedDestinationId":"Save as PDF","version":2}',
    "savefile.default_directory": current_directory  # Set the save directory to the current working directory
}
chrome_options.add_experimental_option("prefs", prefs)
# List of URLs to fetch data from


# Function to login using Chrome
def login():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service,options=chrome_options)
    #driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://www.hackerrank.com/login')
    
    # Enter your login credentials
    
    
    # Find the username and password input fields and enter the credentials
    username_field = driver.find_element(By.XPATH, "//*[@aria-label='Your username or email']")
    username_field.send_keys(username)
    
    password_field = driver.find_element(By.XPATH, "//*[@aria-label='Your password']")
    password_field.send_keys(password)
    
    # Find and click the login button
    login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
    login_button.click()
    
    # Wait for the login process to complete
    driver.implicitly_wait(5)
    
    # Return the logged in driver
    return driver

# Fetch data from each URL after logging in
combined_output = ""
driver = login()
time.sleep(1)
'''
driver.get('https://www.hackerrank.com/submissions/all/100')
data_field = driver.find_element(By.XPATH, "//tbody[@role='rowgroup']")
child_elements = data_field.find_elements(By.XPATH, "./*") 
if len(child_elements) == 0:
    print("The element has no child elements (empty).")
else:
    print(f"The element has {len(child_elements)} child elements.")
'''
#time.sleep(60)
#role="rowgroup"
# Use the Chrome DevTools Protocol to print the page as PDF
# DevTools commands are accessed via driver.execute_cdp_cmd() in Selenium
current_idx=1
main_dict=dict()
for i in range(number_of_pages_hacker_rank-1,-1,-1):
    current_url='https://www.hackerrank.com/submissions/all/'+str(i+1)
    driver.get(current_url)
    data_field = driver.find_element(By.XPATH, "//tbody[@role='rowgroup']")
    child_elements = data_field.find_elements(By.XPATH, "./*") 
    
    if len(child_elements) == 0:
        break
    # Trigger the print command in Chrome
    #row_indicated = driver.find_element(By.XPATH, "//tbody[@role='row']")
    rows = driver.find_elements(By.XPATH, "//tbody[@role='rowgroup']/tr[@role='row']")#
    #child_elements= data_field.find_elements(By.XPATH, "./tr[@role='row']")
    s_idx=len(rows)-1
    for row in reversed(rows):
        
        row = driver.find_elements(By.XPATH, "//tbody[@role='rowgroup']/tr[@role='row']")[s_idx]
        s_idx-=1
        #row=child.find_element(By.XPATH, "//tbody[@role='rowgroup']/tr[@role='row']")
        # Perform actions on each row
        # Find the 5th column in the current row
        
        a_element = row.find_element(By.TAG_NAME, 'a')
        
        a_property = a_element.get_attribute('href')
        columns = row.find_elements(By.TAG_NAME, 'td')
        if len(columns) >= 5 and (columns[1].text.strip().lower() == 'python3' or columns[1].text.strip().lower() == 'pypy3'):
            fifth_column = columns[4]
            if a_property not in main_dict:
                main_dict[a_property]=float(fifth_column.text)
                modified_url = a_property+'/submissions'
               
                driver.get(modified_url)
                time.sleep(1.5)
                internal_cols= driver.find_elements(By.XPATH, "//div[@class='ellipsis submission-result']")#class="table-row-wrapper"
                for internal_col in internal_cols:
                    a_internal_element = internal_col.find_element(By.TAG_NAME, 'a')
                    a_property_internal= a_internal_element.get_attribute('href')
                    driver.get(a_property_internal)
                    screen_code= driver.find_element(By.XPATH, "//div[@class='CodeMirror-cursor']")
                    # Scroll the element into view
                    driver.execute_script("arguments[0].scrollIntoView();", screen_code)

                    # Use ActionChains to perform Ctrl + A and Ctrl + C
                    actions = ActionChains(driver)
                    actions.move_to_element(screen_code)
                    actions.click()
                    actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)
                    actions.key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL)
                    actions.perform()
                    exname=driver.find_element(By.XPATH, "//h1[@class='ui-icon-label page-label']")
                    

                    # Wait a second to ensure the clipboard has the content
                    

                    # Get the copied content from the clipboard using pyperclip
                    copied_text = pyperclip.paste()
                    # Clean up extra newlines
                    cleaned_text = copied_text.replace('\r\n', '\n').replace('\n\n', '\n')
                    with open("scripts.py", "a") as file:
                        file.write('# '+exname.text + '\n')
                        file.write(cleaned_text)
                        file.write("\n")
                        
                        print(exname.text + ' had been saved to scripts.py')
                    break
                driver.get(current_url)
                time.sleep(0.5)
                
             
            
            
        
          # Example action: print the 'href' attribute of the 'a' tag
        #print(row.text)  # Example action: print the text of each row
    
    current_idx+=1
    time.sleep(0.5)
    driver.execute_script('window.print();')
    time.sleep(0.5)
current_directory = os.getcwd()

# Initialize a PdfMerger object
merger = PdfMerger()

# List all PDF files in the current directory
pdf_files = [f for f in os.listdir(current_directory) if f.endswith('.pdf')]

# Sort the files to ensure they are merged in the correct order
pdf_files.sort()

# Merge all the PDF files
for i in range(0,len(pdf_files)):
    if i == 0:
        with open('Submissions _ HackerRank.pdf','rb') as f:
            merger.append(f)
        os.remove('Submissions _ HackerRank.pdf')
    else:
        with open('Submissions _ HackerRank ('+str(i)+').pdf', 'rb') as f:
            merger.append(f)
        os.remove('Submissions _ HackerRank ('+str(i)+').pdf')
# Save the merged PDF to a file
output_filename = "merged_output.pdf"
with open(output_filename, 'wb') as output_file:
    merger.write(output_file)

# Close the merger
merger.close()

print(f"All PDFs merged into {output_filename}")

driver.quit()
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
