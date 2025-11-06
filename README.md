🧭 🚀 Procedimento Diário – Iniciar e Encerrar o Ambiente de Trabalho

🟢 PARTE 1 — INICIAR O TRABALHO
(Tudo começa com o Terminal)
1️⃣ Abrir o Terminal
* Pressione Command (⌘) + Espaço
* Digite Terminal → pressione Enter
Você verá algo como:

Henrique@MacBook-Air ~ %

2️⃣ Ir até a pasta do projeto
cd ~/Documents/pfc1_sinalizacao_horizontal	

Dica: o ~ significa a pasta do seu usuário (como “Documentos”).

3️⃣ Ativar o ambiente virtual Python

source venv/bin/activate
Quando o ambiente estiver ativo, o terminal mostrará algo assim:

(venv) Henrique@MacBook-Air pfc1_sinalizacao_horizontal %
⚠️ Tudo o que você fizer daqui em diante (como rodar scripts Python) será dentro desse ambiente controlado — o lugar certo pra treinar as redes neurais.

4️⃣ Confirmar se o ambiente está funcionando corretamente
Digite:

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
Se aparecer CPU e GPU, está tudo certo:

[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

5️⃣ (Opcional) Abrir o projeto no VS Code

code .
ou manualmente:
* Abre o VS Code
* Vai em File > Open Folder
* Escolhe pfc1_sinalizacao_horizontal
Se preferir usar o Jupyter Notebook, basta digitar:

jupyter notebook
Ele abrirá o navegador com seu ambiente Python interativo.

6️⃣ Rodar seus scripts normalmente
Por exemplo:

python scripts/treino_LBO.py
ou

python scripts/avaliar_modelo.py

🔴 PARTE 2 — ENCERRAR O TRABALHO
Quando terminar seu dia de trabalho ou quiser “fechar tudo” corretamente:
1️⃣ Desativar o ambiente virtual

deactivate
O (venv) vai desaparecer da linha do terminal — isso significa que o ambiente foi desligado com segurança.

2️⃣ Fechar o Jupyter ou VS Code
* No Jupyter: aperte Ctrl + C no terminal onde ele estiver rodando.
* No VS Code: apenas feche a janela (não há problema em deixar arquivos abertos).

3️⃣ Encerrar o Terminal

exit
ou apenas feche a janela.

🧩 Resumão Rápido (para o dia a dia)
Ação	Comando
Entrar no projeto	cd ~/Documents/pfc1_sinalizacao_horizontal
Ativar ambiente	source venv/bin/activate
Testar TensorFlow	python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
Rodar script	python scripts/treino_LBO.py
Sair do ambiente	deactivate
💡 Dica extra: comando “one-liner”
Se quiser agilizar, dá pra ativar o ambiente e abrir o projeto no VS Code de uma vez só:

cd ~/Documents/pfc1_sinalizacao_horizontal && source venv/bin/activate && code .

Quer que eu monte um atalho automático (arquivo .command) pra você dar dois cliques e o Mac já abrir o projeto, ativar o ambiente e deixar tudo pronto pra rodar? Posso gerar esse script personalizado pra tua máquina.










  LNNKK”k zvxekkkkj
