import laser_encoders
eng_encoder = laser_encoders.LaserEncoderPipeline(lang="eng_Latn")
eng_embeddings = eng_encoder.encode_sentences(["Hi!", "This is a sentence encoder. My name is Akash"])
print(eng_embeddings.shape)  # (2, 1024)