from ultralytics import YOLO
import os

# O bloco abaixo é OBRIGATÓRIO no Windows para evitar travamentos no PyTorch
if __name__ == '__main__':
    print("Iniciando o carregamento do modelo YOLOv8 nano...")
    
    # 1. Carregar o modelo pré-treinado
    model = YOLO('yolov8n.pt')
    
    print("Modelo carregado! Iniciando o treinamento...")
    
    # 2. Iniciar o treinamento
    # Caminho ajustado para quem executa a partir da raiz (projeto_medicao)
    caminho_yaml = os.path.abspath('data/raw/data.yaml')
    
    results = model.train(
        data=caminho_yaml,
        epochs=50,          
        imgsz=640,          
        batch=8,           # Reduzido para 8 para não estourar a memória do seu PC
        name='medicao_madeira_v1',
        workers=2          # Limita as threads do processador no Windows para estabilidade
    )

    print("\nTreinamento concluído com sucesso!")
    print("O modelo foi salvo na pasta: runs/detect/medicao_madeira_v1/weights/best.pt")