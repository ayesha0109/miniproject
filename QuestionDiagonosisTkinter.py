
# Importing the libraries
from tkinter import *
from tkinter import messagebox                           
import os            
import webbrowser
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score



class HyperlinkManager:
      
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground="blue", underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)

        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        # add an action to the manager.  returns tags to use in
        # associated text widget
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return

# Importing the dataset
training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

# # Load the disease-specialist mapping (do this once, outside the function ideally)
# disease_specialist_df = pd.read_csv("disease_special.csv")
# disease_to_specialist = dict(zip(disease_specialist_df["Prognosis"], disease_specialist_df["Speciality"]))

# Load and clean datasets (do this once outside the class)
disease_specialist_df = pd.read_csv("disease_special.csv")
disease_specialist_df.columns = disease_specialist_df.columns.str.strip()

special_doctor_df = pd.read_csv("special_doctor.csv")
special_doctor_df.columns = special_doctor_df.columns.str.strip()

# Load the Home Remedies dataset
home_remedies_df = pd.read_csv("Home Remedies.csv")
home_remedies_df.columns = home_remedies_df.columns.str.strip()


# Create mapping from disease to specialist
disease_to_specialist = dict(zip(disease_specialist_df["Prognosis"], disease_specialist_df["Speciality"]))

# ✅ Step 1: Extract symptoms and build relationships
all_symptoms = training_dataset.columns[:-1].tolist()

def get_related_symptoms(initial_symptom, top_n=5):
    if initial_symptom not in all_symptoms:
        return []
    related_symptoms = training_dataset[training_dataset[initial_symptom] == 1][all_symptoms].sum()
    related_symptoms = related_symptoms.sort_values(ascending=False)
    related = [sym for sym in related_symptoms.index if sym != initial_symptom][:top_n]
    return related

# Slicing and Dicing the dataset to separate features from predictions
X = training_dataset.iloc[:, 0:132].values
Y = training_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()

# Encoding String values to integer constants
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(Y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Implementing the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Saving the information of columns
cols     = training_dataset.columns
cols     = cols[:-1]

# Checking the Important features
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Implementing the Visual Tree
from sklearn.tree import _tree

# Method to simulate the working of a Chatbot by extracting and formulating questions
def print_disease(node):
        #print(node)
        node = node[0]
        #print(len(node))
        val  = node.nonzero() 
        #print(val)
        disease = labelencoder.inverse_transform(val[0])
        return disease
def recurse(node, depth):
            global val,ans
            global tree_,feature_name,symptoms_present
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                yield name + " ?"
                
#                ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    yield from recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    yield from recurse(tree_.children_right[node], depth + 1)
            else:
                strData=""
                present_disease = print_disease(tree_.value[node])
#                print( "You may have " +  present_disease )
#                print()
                strData="You may have :" +  str(present_disease)
               
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                
                red_cols = dimensionality_reduction.columns 
                symptoms_given = red_cols[dimensionality_reduction.loc[present_disease].values[0].nonzero()]
#                print("symptoms present  " + str(list(symptoms_present)))
#                print()
                strData="symptoms present:  " + str(list(symptoms_present))
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print("symptoms given "  +  str(list(symptoms_given)) )  
#                print()
                strData="symptoms given: "  +  str(list(symptoms_given))
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
#                print("confidence level is " + str(confidence_level))
#                print()
                strData="confidence level is: " + str(confidence_level)
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print('The model suggests:')
#                print()
                strData='The model suggests:'
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                row = doctors[doctors['disease'] == present_disease[0]]
#                print('Consult ', str(row['name'].values))
#                print()
                strData='Consult '+ str(row['name'].values)
                QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#                print('Visit ', str(row['link'].values))
                #print(present_disease[0])
                hyperlink = HyperlinkManager(QuestionDigonosis.objRef.txtDigonosis)
                strData='Visit '+ str(row['link'].values[0])
                def click1():
                    webbrowser.open_new(str(row['link'].values[0]))
                QuestionDigonosis.objRef.txtDigonosis.insert(INSERT, strData, hyperlink.add(click1))
                #QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
                yield strData
        
def tree_to_code(tree, feature_names):
        global tree_,feature_name,symptoms_present
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []   
#        recurse(0, 1)
    

def execute_bot():
#    print("Please reply with yes/Yes or no/No for the following symptoms")    
    tree_to_code(classifier,cols)



# This section of code to be run after scraping the data

doc_dataset = pd.read_csv('doctors_dataset.csv', names = ['Name', 'Description'])


diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']


doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']




# Execute the bot and see it in Action
#execute_bot()


class QuestionDigonosis(Frame):
    objIter = None
    objRef = None

    def __init__(self, master=None):
        master.title("Question")
        master.state("z")
        QuestionDigonosis.objRef = self
        super().__init__(master=master)
        self["bg"] = "light blue"
        self.createWidget()
        self.iterObj = None

    def createWidget(self):
        default_font = ("Helvetica", 12)
        label_font = ("Helvetica", 12, "bold")
        button_font = ("Helvetica", 12, "bold")
        text_font = ("Helvetica", 12)

        self.lblQuestion = Label(self, text="Question", width=15, bg="#ffe4b5", font=label_font)
        self.lblQuestion.grid(row=0, column=0, rowspan=4, padx=10, pady=5, sticky="n")

        self.lblDigonosis = Label(self, text="Diagnosis", width=15, bg="#ffe4b5", font=label_font)
        self.lblDigonosis.grid(row=4, column=0, sticky="n", pady=5, padx=10)

        self.txtQuestion = Text(self, width=80, height=4, font=text_font, bd=2, relief="groove")
        self.txtQuestion.grid(row=0, column=1, rowspan=4, columnspan=20, pady=5, padx=10)

        self.txtDigonosis = Text(self, width=80, height=14, font=text_font, bd=2, relief="groove")
        self.txtDigonosis.grid(row=4, column=1, columnspan=20, rowspan=20, pady=5, padx=10)

        self.btnNo = Button(self, text="❌ No", width=15, bg="#ffcccc", font=button_font, command=self.btnNo_Click)
        self.btnNo.grid(row=25, column=0, pady=10, padx=10)

        self.btnYes = Button(self, text="✅ Yes", width=15, bg="#ccffcc", font=button_font, command=self.btnYes_Click)
        self.btnYes.grid(row=25, column=1, columnspan=20, sticky="e", padx=10)

        self.btnClear = Button(self, text="🧹 Clear", width=15, bg="#ffe4b5", font=button_font, command=self.btnClear_Click)
        self.btnClear.grid(row=27, column=0, pady=10, padx=10)

        
        self.lblCustomSymptom = Label(self, text="🔍 Enter initial symptom", width=25, bg="#ffe4b5", font=label_font)
        self.lblCustomSymptom.grid(row=28, column=0, pady=10, padx=10)

        self.entrySymptom = Entry(self, width=30, font=default_font)
        self.entrySymptom.grid(row=28, column=1, padx=10)

        self.btnDynamicStart = Button(self, text="🤖 Smart Start", width=15, bg="#b3ffb3", font=button_font, command=self.dynamic_start)
        self.btnDynamicStart.grid(row=29, column=1, pady=5, sticky="e", padx=10)


    
    def btnClear_Click(self):
        self.txtDigonosis.delete(0.0, END)
        self.txtQuestion.delete(0.0, END)

    def btnYes_Click(self):
        global ans
        ans = 'yes'
        try:
            if hasattr(self, 'symptom_flow') and self.current_symptom_index < len(self.symptom_flow):
                self.selected_symptoms.append(self.symptom_flow[self.current_symptom_index])
                self.current_symptom_index += 1
                self.ask_next_symptom()
            else:
                str1 = QuestionDigonosis.objIter.__next__()
                self.txtQuestion.delete(0.0, END)
                self.txtQuestion.insert(END, str1 + "\n")
        except StopIteration:
            pass

    def btnNo_Click(self):
        global ans
        ans = 'no'
        try:
            if hasattr(self, 'symptom_flow') and self.current_symptom_index < len(self.symptom_flow):
                self.current_symptom_index += 1
                self.ask_next_symptom()
            else:
                str1 = QuestionDigonosis.objIter.__next__()
                self.txtQuestion.delete(0.0, END)
                self.txtQuestion.insert(END, str1 + "\n")
        except StopIteration:
            pass

    def dynamic_start(self):
        self.txtDigonosis.delete(0.0, END)
        self.txtQuestion.delete(0.0, END)

        symptom = self.entrySymptom.get().strip().lower().replace(" ", "_")

        if symptom not in all_symptoms:
            messagebox.showerror("Invalid", "Symptom not recognized.")
            return

        related = get_related_symptoms(symptom)

        self.selected_symptoms = [symptom]
        self.current_symptom_index = 0
        self.symptom_flow = related

        self.ask_next_symptom()

    def ask_next_symptom(self):
        if self.current_symptom_index < len(self.symptom_flow):
            question = f"Do you have {self.symptom_flow[self.current_symptom_index].replace('_', ' ')}?"
            self.txtQuestion.delete(0.0, END)
            self.txtQuestion.insert(END, question)
        else:
            self.predict_from_symptoms()
    



    def predict_from_symptoms(self):
    # Generate input vector based on selected symptoms
        input_vector = [1 if symptom in self.selected_symptoms else 0 for symptom in all_symptoms]
    
    # Make prediction
        prediction = classifier.predict([input_vector])[0]
        disease = labelencoder.inverse_transform([prediction])[0]

    # Get specialist for the predicted disease
        specialist = disease_to_specialist.get(disease, "General Physician")

    
        matching_rows = special_doctor_df[special_doctor_df["speciality"] == specialist]

        if not matching_rows.empty:
            selected_row = matching_rows.sample(1).iloc[0]  # Random row
            doctor = selected_row["Doctor's Name"]
            link = selected_row["link"]
        else:
            doctor = "No specialist doctor available"
            link = "N/A"
            

    # Match remedies by symptoms
        remedies = []
        yogas = []
        for symptom in self.selected_symptoms:
            rows = home_remedies_df[home_remedies_df["Health Issue"].str.lower().str.contains(symptom.lower())]
            for _, row in rows.iterrows():
                remedies.append(row["Home Remedy"])
                yogas.append(row["Yogasan"])

    # If no remedies found by symptom, try by disease
        if not remedies:
            rows = home_remedies_df[home_remedies_df["Health Issue"].str.lower().str.contains(disease.lower())]
            for _, row in rows.iterrows():
                remedies.append(row["Home Remedy"])
                yogas.append(row["Yogasan"])

    # If still nothing, give default
        if not remedies:
            remedies = ["No specific home remedies found."]
            yogas = ["No yogasan available."]

    # Display in the GUI
        self.txtDigonosis.insert(END, f"Based on symptoms, you may have: {disease}\n")
        self.txtDigonosis.insert(END, f"Recommended specialist: {specialist}\n")
        self.txtDigonosis.insert(END, f"Suggested Doctor: {doctor}\n")
        self.txtDigonosis.insert(END, f"Link: {link}\n", "blue_link")
        self.txtDigonosis.tag_configure("blue_link", foreground="blue", underline=True)


        self.txtDigonosis.insert(END, f"\n🪴 Home Remedies:\n")
        for r in remedies:
            self.txtDigonosis.insert(END, f"- {r}\n")

        # self.txtDigonosis.insert(END, f"\n🧘 Suggested Yogasans:\n")
        # for y in yogas:
        #     self.txtDigonosis.insert(END, f"- {y}\n")






class MainForm(Frame):
    main_Root = None
    def destroyPackWidget(self, parent):
        for e in parent.pack_slaves():
            e.destroy()
    def __init__(self, master=None):
        MainForm.main_Root = master
        super().__init__(master=master)
        master.geometry("300x250")
        master.title("Account Login")
        self.createWidget()
    def createWidget(self):
        self.lblMsg=Label(self, text="Health Care Chatbot", bg="PeachPuff2", width="300", height="2", font=("Calibri", 13))
        self.lblMsg.pack()
        self.btnLogin=Button(self, text="Login", height="2", width="300", command = self.lblLogin_Click)
        self.btnLogin.pack()
        self.btnRegister=Button(self, text="Register", height="2", width="300", command = self.btnRegister_Click)
        self.btnRegister.pack()
        self.lblTeam=Label(self, text="Made by:", bg="slateblue4", width = "250", height = "1", font=("Calibri", 13))
        self.lblTeam.pack()
        self.lblTeam1=Label(self, text="R.ujwala", bg="RoyalBlue1", width = "250", height = "1", font=("Calibri", 13))
        self.lblTeam1.pack()
        self.lblTeam2=Label(self, text="Noor Ayesha", bg="RoyalBlue2", width = "250", height = "1", font=("Calibri", 13))
        self.lblTeam2.pack()
        self.lblTeam3=Label(self, text="Sathish Sharma", bg="RoyalBlue3", width = "250", height = "1", font=("Calibri", 13))
        self.lblTeam3.pack()
        
    def lblLogin_Click(self):
        self.destroyPackWidget(MainForm.main_Root)
        frmLogin=Login(MainForm.main_Root)
        frmLogin.pack()
    def btnRegister_Click(self):
        self.destroyPackWidget(MainForm.main_Root)
        frmSignUp = SignUp(MainForm.main_Root)
        frmSignUp.pack()



        
class Login(Frame):
    main_Root=None
    def destroyPackWidget(self,parent):
        for e in parent.pack_slaves():
            e.destroy()
    def __init__(self, master=None):
        Login.main_Root=master
        super().__init__(master=master)
        master.title("Login")
        master.geometry("300x250")
        self.createWidget()
    def createWidget(self):
        self.lblMsg=Label(self, text="Please enter details below to login",bg="blue")
        self.lblMsg.pack()
        self.username=Label(self, text="Username * ")
        self.username.pack()
        self.username_verify = StringVar()
        self.username_login_entry = Entry(self, textvariable=self.username_verify)
        self.username_login_entry.pack()
        self.password=Label(self, text="Password * ")
        self.password.pack()
        self.password_verify = StringVar()
        self.password_login_entry = Entry(self, textvariable=self.password_verify, show='*')
        self.password_login_entry.pack()
        self.btnLogin=Button(self, text="Login", width=10, height=1, command=self.btnLogin_Click)
        self.btnLogin.pack()
    def btnLogin_Click(self):
        username1 = self.username_login_entry.get()
        password1 = self.password_login_entry.get()
        
#        messagebox.showinfo("Failure", self.username1+":"+password1)
        list_of_files = os.listdir()
        if username1 in list_of_files:
            file1 = open(username1, "r")
            verify = file1.read().splitlines()
            if password1 in verify:
                messagebox.showinfo("Sucess","Login Sucessful")
                self.destroyPackWidget(Login.main_Root)
                frmQuestion = QuestionDigonosis(Login.main_Root)
                frmQuestion.pack()
            else:
                messagebox.showinfo("Failure", "Login Details are wrong try again")
        else:
            messagebox.showinfo("Failure", "User not found try from another user\n or sign up for new user")


class SignUp(Frame):
    main_Root=None
    print("SignUp Class")
    def destroyPackWidget(self,parent):
        for e in parent.pack_slaves():
            e.destroy()
    def __init__(self, master=None):
        SignUp.main_Root=master
        master.title("Register")
        super().__init__(master=master)
        master.title("Register")
        master.geometry("300x250")
        self.createWidget()
    def createWidget(self):
        self.lblMsg=Label(self, text="Please enter details below", bg="blue")
        self.lblMsg.pack()
        self.username_lable = Label(self, text="Username * ")
        self.username_lable.pack()
        self.username = StringVar()
        self.username_entry = Entry(self, textvariable=self.username)
        self.username_entry.pack()

        self.password_lable = Label(self, text="Password * ")
        self.password_lable.pack()
        self.password = StringVar()
        self.password_entry = Entry(self, textvariable=self.password, show='*')
        self.password_entry.pack()
        self.btnRegister=Button(self, text="Register", width=10, height=1, bg="blue", command=self.register_user)
        self.btnRegister.pack()


    def register_user(self):
        file = open(self.username_entry.get(), "w")
        file.write(self.username_entry.get() + "\n")
        file.write(self.password_entry.get())
        # file.close()
        
        self.destroyPackWidget(SignUp.main_Root)
        
        self.lblSucess=Label(root, text="Registration Success", fg="green", font=("calibri", 11))
        self.lblSucess.pack()
        
        self.btnSucess=Button(root, text="Click Here to proceed", command=self.btnSucess_Click)
        self.btnSucess.pack()
    def btnSucess_Click(self):

        self.destroyPackWidget(SignUp.main_Root)
        frmQuestion = QuestionDigonosis(SignUp.main_Root)

        frmQuestion.pack()



if __name__ == '__main__':
    root = Tk()
    app = MainForm(master=root)
    app.pack()
    root.mainloop()
