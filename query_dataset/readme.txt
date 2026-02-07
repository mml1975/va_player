каталог query_dataset
----------------------

audio_assistant_queries.csv - содержит Request: запрос к модели, Response - ground truth ответ.
                              файл получен соединением файлов после генерации двумя программами
                              query_gen2.py и query_gen_control.py синтетических запросов и ответов                            
                              
gemma_audio_va_result2.json - содержит датасет с набором: текст запроса, ground truth ответ, реальный ответ модели.
                              реальный ответ получен в программе textrq.ipynb

audioplayer_va - содержит датасет с TTS-генерацией двумя голосами (мужской, женский) текстов запроса.
                 в заголовочном файле к нему audioplayer_va.csv содержится:
                 user_text - текст запроса,
                 user_audio - имя wav-файла,
                 assistant_text - gt ответ (из gemma_audio_va_result2.json)
                 генерация производилась в tts_gen.ipynb
