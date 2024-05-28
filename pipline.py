from DoNotAnswer import get_do_not_answer_dataset
import NLLB
import GoogleTranslate
import laser_encoders
from sklearn.metrics.pairwise import cosine_similarity
import os

language_list = ["dan_Latn"]

translation_model = "nllb1.3B"

LLM = "openchat/openchat_3.5"

# Get the do not answer dataset
data_original = get_do_not_answer_dataset()


def translation_pipeline(data, model_name, language,round_trip=True):
    #Translate the questions

    #Copy the data
    data = data.copy().loc[:,["id", "question"]]

    if model_name[:4] == "nllb":
        model = NLLB.nllb_translator(model_name)

        #NLLB needs the language code be in Meta language format
        target_language = language
        source_language = "eng_Latn"

    elif model_name == "Google":
        model = GoogleTranslate.GoogleTranslate()

        #Google translate needs the language code to be in the format two first letters
        target_language = language[:2]
        source_language = "en"

    #Translate forward
    q_list=list(data["question"])
    batch_size=100
    temp_list = []
    for i in range(0, len(q_list), batch_size):
        temp_list.extend(model.translate(q_list[i:i + batch_size], source_language, target_language))
    data.loc[:,"Question translation"] = temp_list
    
    #define the laser encoder
    Laser_encoder_target = laser_encoders.LaserEncoderPipeline(lang=language)
    Laser_encoder_source = laser_encoders.LaserEncoderPipeline(lang="eng_Latn")

    #Encode the questions and the translated questions
    Orignale_embeddings = Laser_encoder_source.encode_sentences(data["question"])
    Tanslated_embeddings = Laser_encoder_target.encode_sentences(data["Question translation"])

    #Calculate the cosine similarity
    data.loc[:,"Cosine"] = [cosine_similarity(Orignale_embeddings[i].reshape(1, -1), Tanslated_embeddings[i].reshape(1, -1))[0][0] for i in range(len(Orignale_embeddings))]

    

    if round_trip:
        #Translate back
        data.loc[:,"Question translation back"] = model.translate(list(data["Question translation"]), target_language, source_language)

        #Encode the translated back questions
        Tanslated_back_embeddings = Laser_encoder_source.encode_sentences(data["Question translation back"])

        #Calculate the cosine similarity
        data.loc[:,"Cosine back"] = [cosine_similarity(Orignale_embeddings[i].reshape(1, -1), Tanslated_back_embeddings[i].reshape(1, -1))[0][0] for i in range(len(Orignale_embeddings))]

    
    #Save the data
    data.to_csv(os.path.join("translations",language[:3]+"_"+model_name+".csv"), index=False)






translation_pipeline(data_original, translation_model, language_list[0])


