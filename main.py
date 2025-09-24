import cv2
import os
import numpy as np
import serial
import time

# Configurações
PORTA_ARDUINO = 'COM8'
BAUDRATE = 9600
TEMPO_LED_ACESO = 10  # Tempo em segundos que o LED ficará aceso após reconhecimento

# Inicializar ligação com Arduino
try:
    arduino = serial.Serial(PORTA_ARDUINO, BAUDRATE, timeout=1)
    print(f"Ligado ao Arduino na porta {PORTA_ARDUINO}")
    time.sleep(2)  # Aguardar inicialização do Arduino
except:
    print(f"Erro ao ligar ao Arduino na porta {PORTA_ARDUINO}")
    arduino = None

# Carregar o classificador facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar reconhecedor facial
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directório com as imagens de treino
directorio_treino = "faces_autorizadas"
directorio_modelo = "modelo"


# Função para treinar o reconhecedor facial
def treinar_reconhecedor():
    if not os.path.exists(directorio_treino):
        os.makedirs(directorio_treino)
        print(f"Directório {directorio_treino} criado. Adicione imagens das pessoas autorizadas.")
        return False

    if not os.path.exists(directorio_modelo):
        os.makedirs(directorio_modelo)

    print("A treinar reconhecedor facial...")

    faces = []
    ids = []
    nomes = {}
    proximo_id = 0

    # Percorrer directórios de pessoas autorizadas
    for pessoa in os.listdir(directorio_treino):
        pasta_pessoa = os.path.join(directorio_treino, pessoa)
        if os.path.isdir(pasta_pessoa):
            nomes[proximo_id] = pessoa
            for ficheiro in os.listdir(pasta_pessoa):
                if ficheiro.endswith(('.jpg', '.jpeg', '.png')):
                    caminho_imagem = os.path.join(pasta_pessoa, ficheiro)
                    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

                    # Detectar faces na imagem
                    face_detectada = face_cascade.detectMultiScale(img, 1.1, 4)

                    for (x, y, w, h) in face_detectada:
                        faces.append(img[y:y + h, x:x + w])
                        ids.append(proximo_id)

            proximo_id += 1

    # Verificar se foram encontradas faces
    if len(faces) == 0:
        print("Nenhuma face encontrada nas imagens de treino.")
        return False

    # Treinar o reconhecedor
    recognizer.train(faces, np.array(ids))

    # Guardar o modelo treinado
    modelo_path = os.path.join(directorio_modelo, "modelo_treinado.yml")
    recognizer.save(modelo_path)

    # Guardar o dicionário de nomes
    import json
    with open(os.path.join(directorio_modelo, "nomes.json"), 'w') as f:
        json.dump(nomes, f)

    print(f"Treino concluído. {len(faces)} faces de {len(nomes)} pessoas.")
    return True


# Função para carregar o modelo treinado
def carregar_modelo():
    modelo_path = os.path.join(directorio_modelo, "modelo_treinado.yml")
    nomes_path = os.path.join(directorio_modelo, "nomes.json")

    if not os.path.exists(modelo_path) or not os.path.exists(nomes_path):
        print("Modelo não encontrado. A iniciar treino...")
        if not treinar_reconhecedor():
            return False, {}

    # Carregar o modelo
    recognizer.read(modelo_path)

    # Carregar o dicionário de nomes
    import json
    with open(nomes_path, 'r') as f:
        nomes = json.load(f)

    # Converter chaves para inteiros (json converte para strings)
    nomes = {int(k): v for k, v in nomes.items()}

    print(f"Modelo carregado. {len(nomes)} pessoas reconhecíveis.")
    return True, nomes


# Função para acender o LED (sem finalizar o sistema)
def acender_led():
    if arduino is not None:
        print(" FACE RECONHECIDA! LED aceso por 5 segundos...")
        arduino.write(b'L')  # Enviar 'L' para acender o LED
        # Criar thread para apagar o LED após o tempo especificado
        import threading
        def apagar_led_depois():
            time.sleep(TEMPO_LED_ACESO)
            arduino.write(b'D')  # Enviar 'D' para apagar o LED
            print("LED apagado.")

        thread_led = threading.Thread(target=apagar_led_depois)
        thread_led.daemon = True
        thread_led.start()
    else:
        print(" SIMULAÇÃO: Face reconhecida! LED aceso por 5 segundos.")


# Função para testar Arduino
def testar_arduino():
    if arduino is not None:
        print("A testar ligação com Arduino...")
        print("A testar LED...")
        arduino.write(b'L')  # Acender LED
        time.sleep(2)
        arduino.write(b'D')  # Apagar LED
        print("Teste de LED concluído.")
    else:
        print("Arduino não está ligado.")


# Menu inicial
while True:
    print("\n=== Sistema de Reconhecimento Facial com LED ===")
    print("1. Iniciar reconhecimento")
    print("2. Treinar novo modelo")
    print("3. Adicionar nova pessoa (captura pela câmara)")
    print("4. Testar Arduino")
    print("5. Sair")

    opcao = input("Escolha uma opção: ")

    if opcao == '1':
        # Carregar o modelo treinado
        modelo_carregado, nomes = carregar_modelo()
        if not modelo_carregado:
            print("Não foi possível carregar ou treinar o modelo. Adicione imagens de treino.")
            continue

        # Inicializar a câmara
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erro ao abrir a câmara.")
            continue

        print("Sistema de reconhecimento facial iniciado.")
        print("A aguardar reconhecimento de face autorizada...")
        print("Prima 'q' para sair.")

        # Configurações de reconhecimento
        limiar_confianca = 70
        pessoa_reconhecida = None
        led_accionado = False
        tempo_reconhecimento = None

        # Ciclo principal de reconhecimento
        while True:
            # Capturar frame da câmara
            ret, frame = video_capture.read()
            if not ret:
                print("Erro ao capturar frame da câmara.")
                break

            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # Desenhar rectângulo
                cor_rectangulo = (0, 255, 0) if pessoa_reconhecida else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), cor_rectangulo, 2)

                if not pessoa_reconhecida:
                    # Reconhecer a face
                    face_roi = gray[y:y + h, x:x + w]
                    id_pessoa, confianca = recognizer.predict(face_roi)

                    # Considerar reconhecido se a confiança for menor que o limiar
                    nome = "Desconhecido"
                    cor = (0, 0, 255)  # Vermelho para desconhecidos

                    if confianca < limiar_confianca and id_pessoa in nomes:
                        nome = nomes[id_pessoa]
                        cor = (0, 255, 0)  # Verde para pessoas autorizadas

                        # Primeira vez que reconhece a pessoa
                        if not led_accionado:
                            print(f" PESSOA AUTORIZADA DETECTADA: {nome}")
                            pessoa_reconhecida = nome
                            led_accionado = True
                            tempo_reconhecimento = time.time()
                            acender_led()

                    # Mostrar nome e confiança
                    texto = f"{nome} ({int(100 - confianca)}%)"
                    cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                else:
                    # Pessoa já foi reconhecida, mostrar o nome confirmado
                    cv2.putText(frame, pessoa_reconhecida, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mostrar estado no ecrã
            if pessoa_reconhecida:
                # Calcular tempo desde o reconhecimento
                tempo_decorrido = int(time.time() - tempo_reconhecimento)
                status_texto = f"PESSOA IDENTIFICADA: {pessoa_reconhecida} (ha {tempo_decorrido}s)"
                cor_status = (0, 255, 0)  # Verde

                # Mostrar instruções para fechar
                cv2.putText(frame, "Prima 'q' para fechar", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                status_texto = "A aguardar reconhecimento..."
                cor_status = (255, 255, 255)  # Branco

            cv2.putText(frame, status_texto, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)

            # Mostrar o resultado
            cv2.imshow('Reconhecimento Facial com LED', frame)

            # Sair se a tecla 'q' for premida
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if pessoa_reconhecida:
                    print(f"🎉 Sistema fechado. Pessoa identificada: {pessoa_reconhecida}")
                else:
                    print("Sistema fechado pelo utilizador.")
                break

        # Libertar recursos
        video_capture.release()
        cv2.destroyAllWindows()

    elif opcao == '2':
        # Treinar novo modelo
        treinar_reconhecedor()

    elif opcao == '3':
        # Adicionar nova pessoa
        nome_pessoa = input("Digite o nome da pessoa: ")
        if not nome_pessoa.strip():
            print("Nome inválido.")
            continue

        # Criar directório para a pessoa se não existir
        pasta_pessoa = os.path.join(directorio_treino, nome_pessoa)
        if not os.path.exists(pasta_pessoa):
            os.makedirs(pasta_pessoa)

        # Inicializar câmara
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erro ao abrir a câmara.")
            continue

        print("\nA capturar imagens para treino. Prima:")
        print("- ESPAÇO para capturar uma imagem")
        print("- ESC para finalizar")

        contador = 0
        while contador < 10:  # Capturar até 10 imagens
            ret, frame = video_capture.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            # Mostrar quantidade de imagens capturadas
            cv2.putText(frame, f"Imagens: {contador}/10", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Detectar face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Captura de Imagens', frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32 and len(faces) > 0:  # ESPAÇO
                # Guardar a imagem se uma face for detectada
                img_path = os.path.join(pasta_pessoa, f"{contador}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Imagem {contador + 1} capturada.")
                contador += 1

        video_capture.release()
        cv2.destroyAllWindows()

        if contador > 0:
            print(f"{contador} imagens capturadas para {nome_pessoa}.")
            print("A iniciar treino com as novas imagens...")
            treinar_reconhecedor()
        else:
            print("Nenhuma imagem capturada.")

    elif opcao == '4':
        # Testar Arduino
        testar_arduino()

    elif opcao == '5':
        if arduino is not None:
            arduino.close()
        print("Sistema encerrado.")
        break

    else:
        print("Opção inválida. Tente novamente.")