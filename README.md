Procedimento para Iniciar e Encerrar o Ambiente de Trabalho

PARTE 1 — INICIAR O TRABALHO
(Tudo começa com o Terminal)

Abrir o Terminal
* Pressione Command (⌘) + Espaço
* Digite Terminal → pressione Enter
Você verá algo como:

Henrique@MacBook-Air ~ %

Ir até a pasta do projeto:

cd ~/Documents/pfc1_sinalizacao_horizontal	

o ~ significa a pasta do seu usuário (como “Documentos”).

Ativar o ambiente virtual Python

source venv/bin/activate

Quando o ambiente estiver ativo, o terminal mostrará algo assim:

(venv) Henrique@MacBook-Air pfc1_sinalizacao_horizontal % 

#Tudo o que você fizer daqui em diante (como rodar scripts Python) será dentro desse ambiente controlado, o lugar certo pra treinar as redes neurais.

Confirmar se o ambiente está funcionando corretamente, digite:

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
Se aparecer CPU e GPU, está tudo certo:

[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

Rodar seus scripts normalmente
Por exemplo:

python scripts/treino_LBO.py

ou

python scripts/avaliar_modelo.py

PARTE 2 — ENCERRAR O TRABALHO
Quando terminar o trabalho ou quiser “fechar tudo” corretamente. Desativar o ambiente virtual:

deactivate

O (venv) vai desaparecer da linha do terminal — isso significa que o ambiente foi desligado com segurança.

Fechar VS Code

No VS Code: apenas feche a janela (não há problema em deixar arquivos abertos).

exit ou apenas feche a janela.



