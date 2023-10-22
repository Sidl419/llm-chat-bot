# llm-chat-bot

Веб-сервис, способный отвечать на вопросы пользователей на основе базы знаний. Для запуска сервиса необходимо скачать в корневую папку файл [модели Llama](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf) и [документы](https://drive.google.com/drive/folders/1RaGIjeWaC8ZNtlfYdHEj-6Vy7L4cHOEE), по которым ведётся поиск (в нашем случае условия Тинькофф Банка), они должны лежать в папке `/tinkoff-terms`.

Ключевые моменты решения:

* Сервис работает на базе CPU.
* База знаний содержится в векторной БД Chroma, эмбеддинги для поиска вычисляются на основе модели `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. При поступлении нового запроса от пользователя сначала достаются ответы из БД, после чего они передаются в контекст модели.
* История диалога с конкретным пользователем хранится в БД Redis по ключу `user_id`.
* Интерфейс реализован через FastAPI.

Для контроля зависимостей в проекте использовался Poetry. Код фоматировался при помощи black.

## Работа с Docker

Контейнер собирается по команде

```[bash]
docker build --tag llm_username:v1 .
```

Сервис можно запустить с помощью следующей команды

```[bash]
docker run --publish 8080:8080 llm_username:v1
```

## Примеры работы

Q: Когда осуществляется начисление процентов на остаток средств

> A: Начисление процентов на остаток средств происходит со дня следующего после даты зачисления денежных средств на счет вклада или даты проlonгации вклада клиента на новый срок. Это определено Тарифным планом и зависит от того, какой тип счета вы используете. Пожалуйста, обратите внимание, что проценты могут быть зачислены не только на счет вклада, но и на накопительный счет. Если у вас есть какие-либо вопросы, не стесняйтесь задавать их мне.

Q: Какие валюты ПС являются рассчетными

> A: Валютные ПС, которые являются рассчетными, определены в разделе 4.15.5. Они включают в себя рубли РФ (расчеты в РФ), рубли РФ (расчеты за пределами РФ), МИР, Visa, MasterCard, UnionPay International. Если операция выполняется в валюте, отличной от валюты Договора кредитной карты и не являющейся одной из расчетных валют ПС, сумма операции конвертируется в расчетную валюту ПС по курсу ПС.

Q: Какой сейчас процент по кредиту?

> A: Добрый день! По вашему запросу, процент по кредиту составляет 24,176% - 33,013% годовых, в зависимости от количества и длины кредитования. Однако, я должен отметить, что это только общая информация, и проценты могут отличаться в зависимости от индивидуальных условий каждого кредитора. Если у вас есть конкретные вопросы или потребности, пожалуйста, не стесняйтесь задавать их мне.

Q: А сколкьо мне будет стоить обслуживание карточки если у меня есть открытый кредит?

> A: Добрый день! Я рад помочь вам. Обслуживание кредитной карты может быть бесплатным или требующим платы, в зависимости от того, какой тип карты вы используете. Если вы имеете открытый кредит, то обслуживание карточки может быть бесплатным. Однако, если вы не уплатите минимальную сумму задолженности по кредитной карте, то вы можете получить санкции, такие как запрет на использование карты или увеличение процентного ставки. Поэтому, я рекомендую уплатить минимальную сумму задолженности в срок, указанном в договоре. Если у вас есть какие-либо вопросы, не стесняйтесь задавать их мне.

Q: Сколько мне будут стоить смски с оповещениями об операциях

> Добрый день! Спасибо за интерес. Стоимость смски с оповещениями об операциях может варьироваться в зависимости от конкретной операции и условий аутентификации. Для получения более точной информации, пожалуйста, просто уточните свой вопрос или обратитесь к нашему специалисту по поддержке клиентов. Мы всегда рады помочь!
