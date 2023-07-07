import torch
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer

# dataset = load_dataset("imdb", split="train")

text = "‘Son, life is like a book. It has numerous chapters. It also has many a lesson in it. It is made up of a wide variety of experiences and resembles a pendulum where success and failure, joy and sorrow are merely extremes of the central reality. The lessons to be learnt from success and failure are equally important. More often than not, failure and sorrow are bigger teachers than success and happiness. You are a cricketer and sportsman. You are fortunate to be representing your country, and that is a great honour. But never forget that this too is just another chapter in the book. Typically, let’s say a person lives for seventy or eighty years or so. How many years will you play sport? Twenty years; if you are very good, maybe even twenty-five years. Even by that yardstick, you will live the majority of your years outside the sphere of professional sport. This clearly means that there is more to life than cricket. I am asking you, son, to keep a pleasant disposition and maintain a balanced nature. Do not allow success to breed arrogance in you. If you remain humble, people will give you love and respect even after you have finished with the game. As a parent, I would be happier hearing people say, “Sachin is a good human being” than “Sachin is a great cricketer” any day.’ My father’s words, which I often heard while growing up, encapsulate my life’s philosophy. I was born to a very close-knit Maharashtrian family in Mumbai’s Bandra East and lived in the Sahitya Sahawas colony, a residential co-operative for writers. I am one of four children, with two brothers and a sister. Nitin, Ajit and Savita are all older than me, and not only am I the youngest in the family but I was also the worst behaved. My father, Ramesh Tendulkar, was an acclaimed Marathi poet, critic and professor, while my mother, Rajani, worked for the Life Insurance Corporation of India. Humility and modesty were their hallmarks and I owe a lot of my personality to my upbringing. Despite all my unreasonableness and all the embarrassments I caused them, my parents never gave up on me. In fact, I have often wondered just how they managed to cope with such a naughty child. Though he must have been pushed to the limits sometimes, my father would never shout at me and was always patient when dealing with my mischief. This added to my respect for my father as I grew older. Losing him during the 1999 World Cup in England remains one of the most traumatic moments of my life and I will forever remain indebted to him for helping me become the human being that I am. My mother, the best cook in the world for me, will do anything to see a smile on my face. She used to make the most delicious fish and prawn curry, baigan bharta and varan bhaat (lentils and rice) for us at home, and I owe my appetite and love of food to her. I fondly remember lying on her lap after eating delicious home-cooked meals, as she sang the most beautiful songs while trying to get me off to sleep. Listening to her while dozing off at the end of the day instilled in me a love for music that has remained with me to this day. My brothers, Nitin and Ajit, have always backed me in my endeavours and, on the cricket side, I owe a lot to Ajit, who is ten years older than me and was a good club cricketer himself but decided to sacrifice his own career to help me achieve my potential. As I said in my farewell speech after my final Test, Ajit and I lived the dream together and he was always my most trusted critic and sounding board. I may have scored the runs, but Ajit was always there with me in spirit, trying to put me right whenever I made a mistake. Even after my last Test innings, we had a discussion about how I had got out and what I had done wrong, despite knowing I’d never play for India again. Ajit is not just my brother, but my closest friend as well. He was always available when I needed him and always put my cricket before his own work. My eldest brother, Nitin, easily the most creative of the siblings, was the strict disciplinarian in the Tendulkar household and helped rein in my exuberance when my mother had almost given up on me. He not only sketches really well, but is also an accomplished writer and poet and has recently written songs for a movie. Nitin, initially a chemistry teacher, subsequently worked for Air India and I remember on one occasion, when I was ten, his flight was delayed and he had to wait at the Centaur (now Sahara Star) hotel in Mumbai. Ajit and I went to have dinner in his room and for the first time in my life I tasted tandoori chicken, which subsequently became one of my favourite dishes. Savita, my sister, gave me my first cricket bat. She travelled to Kashmir for a holiday when I was five and brought me back a Kashmir willow bat. She is easily the calmest of the siblings and has a very reserved and composed demeanour. She stays unruffled in difficult situations and we often consulted her on critical matters while growing up. When she got married, I, not knowing much about rituals and customs, tried to insist that my brother-in-law should come and stay with us rather than Savita having to go away. I did not want to let her go and I must say I missed her terribly when she left home. Never sitting still Undoubtedly I had a fascinating childhood. My early years were never boring; in fact, quite the opposite. I can trace a lot of the stamina and inner strength that sustained me during my cricket career to those early years, which were full of fun. We had moved to Sahitya Sahawas in 1971. In my growing-up years, there was a great deal of construction work taking place there. This gave me and my friends the opportunity to play quite a few pranks on our neighbours. While we were never violent and never caused bodily harm to others, I’m ashamed to admit we sometimes enjoyed having a laugh at the expense of other members of the colony. For us it was fun, plain and simple, but looking back at some of the mischief we got up to now is rather embarrassing. One of our regular tricks was to dig a deep hole in the sand left behind by the contractors and cover it with newspapers before disguising it with sand. Then we’d deliberately lure people to walk over it. As they sank into the crater, we’d be in fits of laughter. Another was to pour water on unsuspecting passers-by from our apartment on the fourth floor, and I remember that feasting on mangoes picked from trees we weren’t supposed to touch was also a favourite pastime. The forbidden nature of the act made it even more compelling and the complaints that would follow did little to put us off. Finally – and this is very embarrassing, looking back now – my friends and I would take pride in locking people in their flats. It wasn’t dangerous, but the resulting delay, which must have caused them immense frustration, seemed very funny at the time. As a child I was first enrolled at the Indian Education Society’s New English School in Bandra. I was a reasonable student and though I was never a class-topper, I did not languish at the bottom either. While school wasn’t altogether boring, the best time of the year was the two-month-long summer break. During the holiday period, I’d hurry down from our apartment at 9 a.m. and would be out in the sun playing for the rest of the day. The domestic help, Lakshmibai, (a common phenomenon in households where both parents were working) would have to bring down my glass of milk and sometimes she would also have to bring out my lunch, because I’d refuse to go up to our apartment. The sweltering heat was never a distraction and I’d be out playing till late in the evening. In fact, even after most of my friends had disappeared to their apartments, I would be out alone trying to amuse myself. There were seven or eight blocks in the colony and sometimes I’d just run around them to expend energy. I’d run seven or eight laps on the trot and do so barefoot. Only when my brother Nitin instructed me to go up would I rush back. I was a little scared of him. He generally didn’t say much to me but when he did it was always the final word. If my mother grew tired of trying to persuade me to come in, she would ask Nitin to perform the task. In our two-bedroom apartment, the four children would all sleep together in one of the bedrooms. I was always the last one to drop off and would keep tossing and turning as the others drifted off. Often, while they’d be lying north–south, I’d end up stretched out east–west, and I’d receive a mouthful when they woke up to find me lying across them. The reprimands were part of the bonding and I never took them to heart. The whole experience brought us closer together. A first taste of Chinese food As a child I loved food. I grew up eating my mother’s wonderful Maharashtrian home cooking and it wasn’t till I was nine years old that I first tried Chinese food. In the early 1980s Chinese cuisine was becoming popular in Mumbai and, having heard so much about it, my colony friends made a plan to go out for a meal together. We each contributed ten rupees – which was a lot of money for me at the time – and I was excited about trying something new. The evening, however, turned out to be a disaster as I paid the price for being one of the youngest in the group. In the restaurant we ordered chicken and sweetcorn soup as a starter. We were sitting at a long table and by the time the soup travelled to me at the far end, there was hardly anything left. The older members of the gang had finished off most of it, leaving very little for us younger ones. The same thing happened with the fried rice and chow mein and I barely managed to get two spoonfuls of each. The older boys had a great evening at our expense but I returned home hungry and thirsty. Dreaming of a bicycle As I kid I could also be quite obstinate. While most of my friends had their own bicycles, I did not and I was determined to have one. My father didn’t really like saying no to me and tried to placate me by saying he’d buy me one in a few weeks. From a financial point of view, it wasn’t easy to bring up four children in Mumbai, but our parents never let us feel any pressure. Not knowing what they had to go through, I remained determined to have my bicycle and refused to go outside and play till I had a new one to show off. It seems a little ridiculous now, but the truth is I didn’t go out to play for a whole week. I just stood on the balcony and sulked and tried to guilt-trip my parents into buying me a bicycle. It was on one of these days that I gave them a real scare. Ours was a fourth-floor apartment with a small balcony with a grille. As a small child, I couldn’t see over the top and, with curiosity often getting the better of me, I would try to get my head through the grille. On this occasion it resulted in disaster. While I succeeded in pushing my head through, I couldn’t get it back in and was stuck there for more than thirty minutes. My parents were flustered to start with, but quickly regained composure. After plenty of oil was squirted on my head, my mother finally pulled me out. Seeing my desperation and worried about what I might get up to next, my father rearranged his finances to buy me a brand-new bicycle. I still don’t know what adjustments he had to make to do so. Nor was I concerned at the time. All I cared about was the bicycle and I immediately showed it off to all my friends. However, my joy was short-lived as I met with a serious accident within hours of getting my precious new bicycle. A fruit and vegetable seller pushing a cart had come to the colony. As we came face to face, I was riding too fast and couldn’t slow down in time. New to the bicycle, I applied the wrong brake and, bang, I hit the cart head on, lost control and was tossed into the air. As I looked down on the world, my only concern was what would happen to my new bicycle. When I came crashing back down, one of the spokes went through the skin just above my right eye. The cut was deep and blood was gushing out of the wound. Far more importantly, my bicycle was badly damaged. News soon reached home that I had hurt myself and my parents were very concerned. I tried to be brave and made out that it was only a minor wound. It wasn’t, and my father had to take me to a plastic surgeon friend of his, who put eight stitches just above the right eye. He gave me a couple of injections and I returned home feeling sorry for myself and frustrated. My mangled bicycle was parked close to our apartment, but my father told me that I wasn’t allowed near it until the wound had healed and that he’d get it repaired in the interim. This time I had to give in, knowing it was the only way I’d get it back. As soon as I’d recovered, I resumed cycling, and within a few months had become an accomplished biker. I could slow-cycle better than most kids and even went on to win a race organized in the colony. I rode with passion and within a few months had developed the ability to slide on one wheel, which took all my friends by surprise. In areas of the colony where there was sand on the concrete, I could get the wheels to slide for ten to fifteen feet, with my body bent at forty-five degrees. I wasn’t bothered about what this was doing to the tyres, of course, as the larger the distance covered, the better I felt. Showing off my skills used to give me a thrill and what added to the fun was that I had learnt these tricks in quick time. Nevertheless, things went wrong sometimes, causing me plenty of embarrassment and pain. In fact, I think I can trace my ability to withstand pain to my exploits as a child. I’d often get cut or hurt but rarely mentioned these minor accidents to anyone at home. So much so that my father got into the habit of examining my body when I was sleeping to check whether I’d injured myself. If he saw me wince in pain, he’d know I’d done something to myself again and he would take me to the doctor the next day. No matter what I’d done, though, my father would never shout or scream at me. More often than not, he’d try to set out the reasons why I should or shouldn’t do certain things, and his explanations left behind a lasting impact. My father’s sense of reason was his biggest virtue and I try to act in the same way with my children. In the wars again I had a lot of adventures as a child, but one that stands out is when I was cut under my eye while playing at Shivaji Park, the breeding ground of cricketers in Mumbai, and had to return home covered in blood. I was captaining my team in a match at Shivaji Park when I was twelve and after our wicketkeeper got injured I asked my team-mates if anyone could keep wicket. No one volunteered and somewhat reluctantly I stepped up to the challenge, even though I’d never tried it before. I was "

values = []
for i in range(13000):
    values.append(text[i:i+1000])

dataset = Dataset.from_dict({"text": values})

model_id = "tiiuae/falcon-7b"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],  # , "word_embeddings", "lm_head"],
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=float(2e-4),
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=10000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

model.config.use_cache = False

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

for name, module in trainer.model.named_modules():
    # if isinstance(module, LoraLayer):
    #     if script_args.bf16:
    #         module = module.to(torch.bfloat16)
    if "norm" in name:
        module = module.to(torch.float32)
    # if "lm_head" in name or "embed_tokens" in name:
    #     if hasattr(module, "weight"):
    #         if script_args.bf16 and module.weight.dtype == torch.float32:
    #             module = module.to(torch.bfloat16)

trainer.train()
