import tkinter as tk
import tkinter.font as font

def reciever(SStr, ppp):
    window = tk.Tk()
    window.title("Receiver(Patient)")
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window_width = int(screen_width * 0.5) 
    window_height = int(screen_height * 0.5) 
    window.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")
    
    label_font = font.Font(family='Arial', size=12, weight='bold')
    label = tk.Label(window, text="The predicted disease is:", font=label_font)
    label.pack(pady=10)
    
    text_font = font.Font(family='Arial', size=12)
    text = tk.Label(window, text=SStr, font=text_font)
    text.pack()

    scrollbar = tk.Scrollbar(window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    description_font = font.Font(family='Arial', size=12)
    description = tk.Text(window, wrap="word", yscrollcommand=scrollbar.set, font=description_font)
    if SStr=="Measles":
        ppp="Treatment:\n1.Supportive Care: There is no specific antiviral treatment for measles. Treatment focuses on relieving symptoms and supporting the immune system's response to the virus.\n2.Rest and Hydration: Adequate rest and hydration are crucial for managing fever and promoting recovery. Patients are advised to drink plenty of fluids to prevent dehydration.\n3.Fever Reduction: Over-the-counter medications such as acetaminophen or ibuprofen can help reduce fever and alleviate discomfort. Aspirin should be avoided in children due to the risk of Reye's syndrome.\n4.Vitamin A Supplementation: In areas where vitamin A deficiency is prevalent, supplementation with vitamin A has been shown to reduce the severity of measles and decrease the risk of complications, particularly in children.\n\nRecovery Steps:\n1.Isolation: Since measles is highly contagious, infected individuals should be isolated to prevent the spread of the virus to others, especially those who are unvaccinated or immunocompromised.\n2.Quarantine: Close contacts of measles cases, particularly those who are unvaccinated or have weakened immune systems, may need to be quarantined to prevent transmission.\n3.Follow-Up: Patients recovering from measles should undergo follow-up evaluation with a healthcare provider to monitor their symptoms and ensure proper recovery.\n4.Vaccination: The measles vaccine, typically administered as part of the measles, mumps, and rubella (MMR) vaccine series, is highly effective in preventing measles infection. Vaccination is the most effective way to protect against measles and its complications.\n5.Complication Monitoring: While most cases of measles resolve without complications, some individuals, especially young children and immunocompromised individuals, may develop complications such as pneumonia, encephalitis, or otitis media. Close monitoring and prompt medical attention are essential if complications arise."
    elif SStr=="Chickenpox":
        ppp="Treatment:\n1.Supportive Care: There is no specific antiviral treatment for chickenpox. Treatment primarily focuses on relieving symptoms and promoting comfort.\n2.Itch Relief: Over-the-counter antihistamines, such as diphenhydramine, can help alleviate itching associated with the chickenpox rash. Calamine lotion or oatmeal baths may also provide relief.\n3.Fever Management: Acetaminophen or ibuprofen can be used to reduce fever and alleviate discomfort. Aspirin should be avoided in children and teenagers due to the risk of Reye's syndrome.\n4.Hydration: Encouraging adequate fluid intake is essential to prevent dehydration, especially in children with chickenpox who may be reluctant to eat or drink due to discomfort.\n\nRecovery Steps:\n1.Isolation: Individuals with chickenpox should be isolated from others, particularly those who are at high risk of complications, such as pregnant women, newborns, and individuals with weakened immune systems.\n2.Quarantine: Close contacts of chickenpox cases who are susceptible to the virus, such as unvaccinated individuals or those who have not had chickenpox previously, may need to be quarantined to prevent transmission.\n3.Monitoring: Patients recovering from chickenpox should be monitored for complications, such as bacterial skin infections, pneumonia, or encephalitis, especially if they are at increased risk.\n4.Preventive Measures: Vaccination is the most effective way to prevent chickenpox and its complications. The varicella vaccine, typically administered as part of the routine childhood immunization schedule, provides long-term immunity against the virus."
    elif SStr=="Monkeypox":
        ppp="Treatment:\n1.Supportive Care: There is no specific antiviral treatment for monkeypox. Treatment primarily focuses on supportive care to alleviate symptoms and promote recovery.\n2.Pain Management: Over-the-counter pain relievers such as acetaminophen or ibuprofen can help reduce fever and alleviate muscle aches and discomfort associated with monkeypox.\n3.Hydration: Maintaining adequate hydration is essential, especially if fever and sweating are causing fluid loss. Patients should drink plenty of fluids to prevent dehydration.\n4.Rest: Adequate rest is crucial for supporting the immune system's response to the virus and facilitating recovery. Patients should rest and avoid strenuous activities until symptoms improve.\n\nRecovery Steps:\n1.Isolation: Similar to other contagious viral diseases, individuals diagnosed with monkeypox should be isolated to prevent transmission to others. Isolation precautions help reduce the risk of secondary cases and contain outbreaks.\n2.Quarantine of Contacts: Close contacts of monkeypox cases, particularly those who have had direct contact with the patient's skin lesions or body fluids, may need to be quarantined and monitored for signs of infection.\n3.Monitoring and Follow-Up: Patients recovering from monkeypox should undergo regular follow-up evaluations with healthcare providers to monitor their symptoms, assess their progress, and ensure appropriate medical management.\n4.Preventive Measures: Preventive measures, such as hand hygiene, respiratory etiquette, and avoiding close contact with sick individuals, can help reduce the risk of monkeypox transmission.\n5.Vaccination: While there is currently no specific vaccine for monkeypox available for widespread use, research into monkeypox vaccines is ongoing. Vaccination efforts may be considered in outbreak settings or for individuals at high risk of exposure, such as healthcare workers."
    elif SStr=="Healthy":
        ppp="Skin is Predicted as Healthy :)\n\nCongratulations! Your skin appears to be healthy. Maintaining healthy skin is essential for overall well-being. Here are some tips to keep your skin in good condition:\n\n1.Regular Cleansing: Cleanse your skin daily with a gentle cleanser to remove dirt, oil, and impurities.\n2.Hydration: Drink plenty of water to stay hydrated, which helps keep your skin moisturized from within.\n3.Sun Protection: Use sunscreen with broad-spectrum protection and a high SPF to shield your skin from harmful UV rays.\n4.Healthy Diet: Eat a balanced diet rich in fruits, vegetables, and whole grains to nourish your skin from the inside out.\n5.Moisturization: Apply a moisturizer suitable for your skin type to keep it hydrated and supple.\n6.Avoid Harsh Products: Avoid using harsh chemicals or abrasive products that can irritate your skin and disrupt its natural balance.\n\nRemember to consult a dermatologist if you have any concerns about your skin health or notice any changes or abnormalities."
    description.insert(tk.END, ppp)
    description.pack(fill="both", expand=True)

    scrollbar.config(command=description.yview)

    window.mainloop()

# Example usage:
disease_predicted = "Monkeypox"
disease_description = ""
reciever(disease_predicted, disease_description)
