# SFX Tagger

Ferramenta local para preencher metadados de bibliotecas de efeitos sonoros.
A IA ouve cada som e sugere **CatID UCS**, **FXName**, **Description** e **Keywords**.
Tu revês, corriges e gravas os metadados **dentro dos próprios WAV**, nos formatos padrão
(BWF, iXML e RIFF INFO) que as bases de dados de SFX e os DAW leem.

Tudo corre no teu computador: os sons não são enviados para lado nenhum.

---

## Instalação (só uma vez)

Precisas de cerca de **3 GB livres** e de internet durante a instalação.

### Windows
1. Instala o **Python 3.12** em <https://www.python.org/downloads/>.
   Durante a instalação, marca a opção **"Add python.exe to PATH"**.
2. Abre a pasta `SFXTagger` e faz duplo clique em **`instalar_windows.bat`**.
3. Espera até aparecer *"Instalação concluída"*. Na primeira vez demora alguns minutos,
   porque descarrega os modelos de IA (cerca de 1,4 GB).

### macOS
1. Instala o **Python 3.12** em <https://www.python.org/downloads/>.
2. Abre o **Terminal**, escreve `bash ` (com um espaço no fim), arrasta o ficheiro
   **`instalar_mac.command`** para a janela do Terminal e carrega Enter.
3. Espera até aparecer *"Instalação concluída"*. A partir daí, o `iniciar_mac.command`
   abre com um duplo clique.

> Funciona em Macs com chip Apple (M1–M4) e com chip Intel. Nos Intel, o instalador
> escolhe sozinho versões compatíveis, mas precisa do Python 3.10–3.12.
> Se o macOS bloquear um ficheiro `.command`, usa botão direito → **Abrir** → **Abrir**.

## Abrir a app

- **Windows:** duplo clique em `iniciar_windows.bat`
- **macOS:** `bash iniciar_mac.command` no Terminal (ou botão direito → Abrir)

A app abre no browser em `http://127.0.0.1:8765`. Deixa a janela preta (consola) aberta
enquanto trabalhas e fecha-a quando terminares.

---

## Como trabalhar

1. **Carregar os sons**, de uma de duas formas:
   - **Arrasta os ficheiros para a janela** (ou usa **Carregar sons**). Aceita WAV, AIFF,
     FLAC, OGG e MP3, e os que não são WAV são convertidos para WAV 24-bit. Os sons ficam numa
     cópia em `Documentos/SFX Tagger/` e os teus originais não são tocados. A análise começa sozinha.
   - **Abrir pasta**: trabalha diretamente numa pasta que já tenhas. As subpastas também são incluídas.
2. **Biblioteca**: preenche uma vez os campos comuns a toda a biblioteca:
   Library, Designer, CreatorID (as tuas iniciais), SourceID (nome do projeto), Microfone…
3. **Analisar**: a IA ouve todos os sons. Conta com cerca de 0,5–1 s por som.
4. **Revê** som a som. O som toca automaticamente quando o escolhes.
   - Escolhe a categoria certa entre as 5 sugestões (teclas **1**–**5**) ou pesquisa no
     campo CatID (ex.: *"door wood"*, *"whoosh"*, *"footstep"*).
   - Ajusta o FXName, a Description e as Keywords. Clica nas sugestões `+ keyword` para as acrescentar.
   - **O que ouves?**: escreve em poucas palavras o que o som é realmente (ex.: *iron gate
     slam*) e carrega Enter. A IA volta a ordenar as sugestões com essa pista.
   - **Ctrl+Enter** (Confirmar e seguinte) confirma o som e passa ao seguinte.
5. **Gravar nos ficheiros**: escreve os metadados nos WAV. Opcionalmente renomeia os
   ficheiros para o padrão UCS `CatID_FXName_CreatorID_SourceID.wav`.
6. **Exportar**:
   - **Metadados (CSV)**: folha de cálculo com todos os campos.
   - **Relatório da IA (CSV)**: para cada som confirmado, o que a IA sugeriu, o que tu
     escreveste que ouvias, a categoria final e se a IA acertou. Envia-o a quem está a
     afinar a ferramenta.
   - **Sons com metadados (ZIP)**: descarrega os WAV já etiquetados. É útil sobretudo para os sons
     carregados por arrastar.

### Ditado por voz
Ao lado de **Description**, **Location**, **Notes** e **O que ouves?** há um botão de
microfone. Clica, descreve o som, o ambiente e o local, e clica de novo para terminar.
O texto aparece no campo.
- Podes falar em português. Por omissão, a Description, o FXName, as Keywords e o "O que ouves?"
  são **traduzidos para inglês** automaticamente; Location e Notes ficam como disseste.
  Isto muda-se em **Biblioteca**.
- O reconhecimento de voz (Whisper) corre no teu computador, por isso a voz não sai daí. Na primeira
  utilização descarrega um modelo de cerca de 1 GB.
- O browser pede autorização para usar o microfone na primeira vez.

### A IA aprende com as vossas confirmações
Cada som confirmado ("Confirmar e seguinte") passa a ser um **exemplo**: a impressão
digital do som (sem o áudio) mais a categoria certa. A partir daí, sons parecidos recebem
essa categoria como sugestão, e o painel do som indica *"Aprendido com sons já confirmados"*.

**Partilhar com colegas** (botão **Conhecimento**):
1. Cada pessoa carrega em **Exportar o meu conhecimento** e obtém um ficheiro `.sfxk`
   pequeno (cerca de 2 KB por som).
2. Envia-o a quem quiser (email, WhatsApp, Drive…).
3. Quem recebe carrega em **Importar conhecimento de um colega…** e depois em
   **Atualizar as sugestões dos sons por rever**.

O painel mostra quantos exemplos há, quem contribuiu, a percentagem de acerto da IA e
as categorias que ela mais confunde. Os exemplos de uma pessoa podem ser removidos.
Preenche o teu **CreatorID** em "Biblioteca" para os teus exemplos ficarem com o teu nome.

### Quão certa é a IA?
Medido em 800 gravações reais etiquetadas (benchmark ESC-50 em categorias UCS, sem pistas
no nome do ficheiro):
- a 1.ª sugestão está certa **77%** das vezes, e a categoria certa está entre as 5 sugestões **91%** das vezes;
- com sons já confirmados na memória, sons parecidos chegam a **83%** com 1 exemplo por tipo de som,
  e a **91%** com 5 exemplos.

Nomes de ficheiro descritivos (`porta_madeira_ranger.wav`) ajudam ainda mais. Os detalhes estão em
`ferramentas/benchmark`.

### Precisão da IA
Cada vez que confirmas um som, a app compara a tua categoria final com a 1.ª sugestão
original da IA. Por cima da lista aparece o resultado, por exemplo *"IA acertou 18/25 (72%)"*,
e cada som mostra *IA acertou*, *estava nas sugestões* ou *IA errou*.

### Trabalhar em lote
- **Selecionar série**: escolhe os outros takes do mesmo nome (`Porta 01`, `Porta 02`…).
- **Selecionar semelhantes**: a IA encontra sons parecidos com o que está aberto.
- **Editar selecionados…**: copia campos (CatID, Keywords, Notes…) para todos os selecionados.
- Na lista: **Shift+clique** seleciona um intervalo e **Ctrl/Cmd+clique** junta ou retira um som.

### Atalhos de teclado
| Tecla | Ação |
|---|---|
| Setas para cima / baixo | som anterior / seguinte |
| Espaço | tocar / parar |
| 1–5 | escolher a sugestão de categoria |
| Ctrl+Enter (Cmd+Enter no Mac) | confirmar e seguinte |
| / | procurar na lista |
| Esc | sair do campo de texto |

### Cores na lista
amarelo = por rever (sugestão da IA) · azul = revisto · verde = gravado no ficheiro ·
vermelho = erro · ponto laranja antes do nome = alterações ainda por gravar

---

## O que é gravado nos WAV

| Bloco | Campos | Quem lê |
|---|---|---|
| **iXML** `<USER>` | CATID, CATEGORY, SUBCATEGORY, CATEGORYFULL, FXNAME, DESCRIPTION, KEYWORDS, NOTES, LIBRARY, DESIGNER, MICROPHONE, RECMEDIUM, LOCATION, CREATORID, SOURCEID | bases de dados de SFX com suporte UCS/iXML (ex.: BaseHead, Soundminer) |
| **BWF bext** | Description (até 256 caracteres, sem acentos), Originator = Designer | praticamente todos os DAW |
| **RIFF INFO** | INAM (FXName), ICMT (Description), IKEY (Keywords), IGNR (Categoria), IART (Designer), IPRD (Library) | Reaper, Windows, ffmpeg |

- O **áudio nunca é alterado**: só os blocos de metadados são reescritos (foi testado bit a bit).
- Com a opção de cópia de segurança ativa, o original de cada ficheiro fica guardado em
  `<pasta>/.sfxtagger/backup/` na primeira vez que a app lhe mexe.
- O progresso (sugestões, edições, estado) fica em `<pasta>/.sfxtagger/projeto.json`.
  Podes fechar a app e continuar mais tarde.

## Dicas para melhores sugestões
- **Bons nomes ajudam muito.** Um ficheiro `porta_madeira_ranger_03.wav` dá melhores
  sugestões do que `ZOOM0001.wav`. A IA combina o som com as palavras do nome,
  de preferência em inglês.
- A IA é muito boa a escolher a **família** do som (portas, vidro, whooshes, chuva). Já não
  sabe o **contexto**: que porta, onde foi gravada, com que microfone. Essa parte é tua.
- Os sons desenhados ou abstratos (UI, sci-fi, magia) são os mais difíceis. Revê-os com atenção.

## Limitações atuais
- Só WAV (incluindo BWF). Ficheiros RF64 com mais de 4 GB ainda não são suportados.
- Não lê nem escreve o bloco proprietário do Soundminer (SMED). Confirma na ferramenta que
  usas se ela importa os campos iXML/UCS (a exportação em CSV serve de alternativa).
- A IA não escreve frases ricas na Description. Deixa um rascunho factual para completares.

## Problemas comuns
- **"Não encontrei o Python"**: reinstala o Python e marca a opção "Add to PATH" (Windows).
- **A app não abre no browser**: abre manualmente `http://127.0.0.1:8765`.
- **Um som não toca**: alguns formatos (multicanal, 192 kHz) são convertidos para pré-escuta.
  Na primeira vez pode demorar um pouco.

---

Categorias: Universal Category System (UCS) v8.2, iniciativa de domínio público,
<https://universalcategorysystem.com>. Modelo de IA: LAION CLAP (`laion/larger_clap_general`).
