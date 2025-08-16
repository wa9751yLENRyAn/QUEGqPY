"""
A module for generating illustrations for Anki cards using LLMs and image generation models.

This module provides functionality to automatically create relevant illustrations 
for Anki cards using large language models to generate prompts, which are then
used with image generation models like DALL-E or Stable Diffusion.
"""
from inspect import signature
from rapidfuzz.fuzz import ratio as levratio
from textwrap import dedent
import base64
from bs4 import BeautifulSoup
import random
import json
import warnings
import io
import requests
import os
import datetime
import fire
import re
import time
from pathlib import Path
from joblib import Memory
from typing import List, Dict, Tuple
import copy

import hashlib
from tqdm import tqdm
from PIL import Image

import litellm
from litellm import completion, image_generation

from utils.misc import send_ntfy, load_formatting_funcs, replace_media
from utils.llm import load_api_keys, llm_price, sd_price, tkn_len, chat, model_name_matcher
from utils.anki import anki, addtags, removetags, sync_anki, updatenote
from utils.logger import create_loggers
from utils.datasets import load_dataset, load_and_embed_anchors, filter_anchors, semantic_prompt_filtering

Path("databases").mkdir(exist_ok=True)
ILLUSTRATOR_DIR = Path("databases/illustrator")
ILLUSTRATOR_DIR.mkdir(exist_ok=True)

api_keys = load_api_keys()

d = datetime.datetime.today()
today = f"{d.day:02d}_{d.month:02d}_{d.year:04d}"
log_file = ILLUSTRATOR_DIR / f"{today}.logs.txt"
Path(log_file).touch()
whi, yel, red = create_loggers(log_file, ["white", "yellow", "red"])

try:
    from stability_sdk import client
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
    # make stability_sdk optional if the user has trouble installing the lib and needs only Dall-E
except Exception as err:
    red(f"Error when importing stability_sdk, you won't be able to use stable diffusion: '{err}'")

# The Major System is a mnemonic system for memorizing numbers by converting them
# to consonant sounds, then into words. This table maps digits to their corresponding
# consonant sounds to help create memorable associations.
major_table = {
    0: "S like in saw, sea, sew",
    1: "T like in thai, tea, tie, tow",
    2: "N like in no, new",
    3: "M like in mow, my, meh",
    4: "R like in raw, rye, ray",
    5: "L like in low, lee, lie",
    6: "Sshh like in shy, shiah, sha",
    7: "K sound like in cow, queue, caw",
    8: "F like in fire, faux, few",
    9: "P like in pie, pew, pay, pee",
}

roman_numerals = {
    "I": "1",
    "II": "2",
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
}

mem = Memory(".cache", verbose=False)

@mem.cache
def cached_image_generation(*args, **kwargs):
    """
    Cached wrapper around litellm's image_generation function.
    
    Uses joblib.Memory to cache results to avoid regenerating identical images.
    
    Parameters
    ----------
    *args
        Positional arguments passed to image_generation
    **kwargs
        Keyword arguments passed to image_generation
        
    Returns
    -------
    dict
        The image generation result from litellm
    """
    return image_generation(*args, **kwargs)


class AnkiIllustrator:
    VERSION = "2.5"
    SEED = 42 ** 5  # keeping the seed constant probably helps with
    # character consistency

    def __init__(
        self,
        query: str = "(rated:2:1 OR rated:2:2 OR tag:AnkiIllustrator::todo OR tag:AnkiIllustrator::FAILED) -tag:AnkiIllustrator::to_keep -is:suspended -body:*img*",
        field_names: List[str] = None,
        n_image: int = 1,
        sd_steps: int = 100,
        n_note_limit: int = 500,
        string_formatting: str = None,
        memory_anchors_file: str = None,
        major_system: bool = True,  # Enable Major System hints - adds phonetic sound mappings for digits to help create mnemonics
        llm_max_token: int = 5000,
        dataset_path: str = None,
        dataset_sanitize_path: str = None,
        max_sanitize_trial: int = 4,
        # llm_model: str = "openai/gpt-4o",
        # llm_model: str = "anthropic/claude-3-5-sonnet-20240620",
        llm_model: str = "openrouter/anthropic/claude-3.5-sonnet:beta",
        # embedding_model: str = "mistral/mistral-embed",
        embedding_model: str = "openai/text-embedding-3-small",
        image_model: str = "openai/dall-e-3",
        do_sync: bool = True,  # don't sync by default because it can impede card creation otherwise
        ntfy_url: str = None,
        disable_notif: bool = False,
        open_browser: bool = False,
        debug: bool = False,
        force: bool = False,
    ):
        """
        Parameters
        ----------
        query: str, default to "(rated:2:1 OR rated:2:2 OR tag:AnkiIllustrator::todo OR tag:AnkiIllustrator::FAILED) -tag:AnkiIllustrator::to_keep -is:suspended",

        field_names: List[str], default None
            list (or comma separated string) of the field of the note to load
            and give to the LLM as prompt.

        n_image: int, default 1
            number of image to generate

        sd_steps: int, default 100
            precision of stable diffusion

        n_note_limit: int
            If number of note to do is is more than this
            many: crop to that many to reduce cost

        string_formatting: str, default None
            path to a python file declaring functions to specify specific
            formatting.

            In illustrator, functions that can be loaded are:
            - "cloze_input_parser"
            it mist take a unique string argument and return a unique string.

            They will be called to modify the note content before sending
            to the LLM

        dataset_path: str
            path to a file with ---- separated messages (including system
            prompt) showing the reasonning to create illustrations

        dataset_sanitize_path: str
            path to a file with ---- separated messages (including system
            prompt) showing how to correct the offensive or graphic
            message to avoid refusals by the image generator

        max_sanitize_trial: int, default 4
            max number of trial to sanitize input for avoiding images that are
            too grahical

        memory_anchors_file: str
            path to a file containing memory anchors in the form of a valid
            json file, keys are the notion and value is the anchor.
            Embeddings are used to match the closest notions to each note

        major_system: bool, default True
            if True, will detect digits in the cloze and mention a helpful
            help to the prompt for the LLM to use the major system in the
            mnemonics

        llm_max_token: int, default 5000
            max number of token when asking the LLM for a prompt

        llm_model: str, default "anthropic/claude-3-5-sonnet-20240620"
            support any model supported by litellm

        embedding_model: str, default "openai/text-embedding-3-small"
            embedding model to use, in litellm format

        image_model: str, default "openai/dall-e-3",
            either dall-e-3 or StableDiffusion

        do_sync: bool, default True
            if True: will trigger an anki sync on start and finish

        ntfy_url: str, default None
            url to use with ntfy.sh to send the status updates

        disable_notif: bool, default False
            if True, won't send notification to phone via ntfy.sh

        open_browser: bool, default False
            If True, open anki's browser at the end of the run. If 'always'
            will refresh the browsing window for each new processed note.

        force: bool, default False
            if True, will not ignore note that already contain an
            illustration of the same version.
            Used for debugging, resetting or if you're rich.
        """
        litellm.set_verbose = debug
        self.hisf = ILLUSTRATOR_DIR
        self.rseed = int(random.random() * 100000)
        whi(f"Random seed: '{self.rseed}'")
        d = datetime.datetime.today()
        if d.hour <= 5:
            # get yesterday's date if it's too early in the day
            d = datetime.datetime.today() - datetime.timedelta(1)
        self.today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
        # logger for tqdm progress bars
        self.sd_steps = sd_steps
        self.disable_notif = disable_notif
        self.n_image = n_image
        self.major_system = major_system

        if isinstance(field_names, list):
            assert not any("," in f for f in field_names), (
                "Detected a list of field_names where one contains a comma")
        else:
            assert isinstance(field_names, str)
            field_names = field_names.split(",")
        self.field_names = field_names

        # load user_anchors
        self.anchors = {}
        if memory_anchors_file:
            self.anchors, self.embeddings = load_and_embed_anchors(
                path=memory_anchors_file,
                model=embedding_model,
            )

        self.llm_max_token = llm_max_token
        self.dataset = load_dataset(dataset_path)
        self.dataset[0]["content"] = self.dataset[0]["content"].replace("MODEL", image_model)
        self.dataset_sanitize = load_dataset(dataset_sanitize_path, check_args={"must_be_unique": False})
        self.max_sanitize_trial = max_sanitize_trial

        # remove phonetic or anchors from dataset if needed
        for i, d in enumerate(self.dataset):
            if i == 0:
                continue
            lines = [l.strip() for l in d["content"].splitlines() if l.strip()]
            if not major_system:
                lines = [li for li in lines if not li.startswith("Phonetic:")]
            if not self.anchors:
                lines = [li for li in lines if not li.startswith("Anchors:")]
            con = "\n".join(lines).strip()
            self.dataset[i]["content"] = con

        if string_formatting is not None:
            red(f"Loading specific string formatting from {string_formatting}")
            cloze_input_parser = load_formatting_funcs(
                    path=string_formatting,
                    func_names=["cloze_input_parser"]
            )[0]
            for func in [cloze_input_parser]:
                params = dict(signature(func).parameters)
                assert len(params.keys()) == 1, f"Expected 1 argument for {func}"
                assert "cloze" in params, f"{func} must have 'cloze' as argument"
            self.cloze_input_parser = cloze_input_parser
        self.string_formatting = string_formatting

        # format browser query
        self.original_query = query
        if not force:
            # only if illustrator has not been updated
            query += f" -AnkiIllustrator:*VERSION:{self.VERSION}* "
        else:
            red("--force enabled, this will not ignore cards with illustration")

        # sync first
        if do_sync:
            sync_anki()

        # load api keys and price etc
        assert "OPENAI_API_KEY" in os.environ or "STABLEDIFFUSION_API_KEY" in os.environ, (
            "Missing either openai or stablediffusion api keys")

        if llm_model in llm_price:
            self.llm_price = llm_price[llm_model]
        elif llm_model.split("/", 1)[1] in llm_price:
            self.llm_price = llm_price[llm_model.split("/", 1)[1]]
        elif model_name_matcher(llm_model) in llm_price:
            self.price = llm_price[model_name_matcher(llm_model)]
        else:
            raise Exception(f"{llm_model} not found in llm_price")
        assert image_model in ["StableDiffusion",
                               "openai/dall-e-3"], "invalid image_model"
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.image_model = image_model

        if image_model == "StableDiffusion":
            # load stable diffusion and misc configs
            # pricing has to change if any parameter changes
            self.img_price = sd_price[str(self.sd_steps)]
            self.sd_cfg_scale = 9
            self.sd_apikey = os.environ["STABLEDIFFUSION_API_KEY"]
            self.stabapi = self._load_stable_diffusion()
        else:
            # DALL·E 3	1024×1024	$0.040 / image
            #           1024×1792, 1792×1024	$0.080 / image
            self.img_price = 0.04

        # compute total price so far and tell the user
        self._total_cost()

        # find nid of the notes
        red(f"Loading notes with query '{query}'")
        notes = anki(action="findNotes", query=query)
        if not notes:
            raise Exception("No notes corresponding to query found")
        yel(f"Found '{len(notes)}' notes")

        # limit number of notes if too many
        if len(notes) > n_note_limit:
            yel(f"Limiting to only {n_note_limit}")
        notes = notes[:n_note_limit]

        # gather info about those notes
        notes_info = anki(action="notesInfo", notes=notes)
        assert len(notes_info) == len(notes), "Invalid notes info length"
        self.notes_info = notes_info

        # check that no media re present in the main field
        for note in notes_info:
            for field_name in self.field_names:
                _, media = replace_media(
                    content=note["fields"][field_name]["value"],
                    media=None,
                    mode="remove_media")
                assert not media, f"Found media '{media}' in {note}"

        # filter notes
        self._extract_content()

        # loop over notes
        print("\n\n")
        time.sleep(5)
        for i, note in tqdm(
            enumerate(self.notes_info),
            desc="Processing notes",
            unit="notes",
            total=len(self.notes_info),
        ):
            try:
                # create image prompt from anki card
                input_token_cost, output_token_cost, image_prompt, reasonning, discarded = (
                    self._create_prompt(note_info=note)
                )

                # create image from image prompt
                imgs_dict = []
                if self.image_model == "StableDiffusion":
                    for n in tqdm(
                        range(n_image),
                        desc="Creating image from prompt",
                        unit="image",
                    ):
                        newimage = self._create_image_sd(
                            prompt=image_prompt,
                            seed=self.rseed * (n + 1),
                            cfg_scale=self.sd_cfg_scale - n,
                        )
                        imgs_dict.append(newimage)
                else:
                    imgs_dict.extend(
                        self._create_image_dalle(prompt=image_prompt))
                    if "san_input_token_cost" in imgs_dict[-1]:
                        # if family friendly filter is triggered, a few more
                        # llm calls are made
                        input_token_cost += imgs_dict[-1]["san_input_token_cost"]
                        output_token_cost += imgs_dict[-1]["san_output_token_cost"]

                # compute cost
                img_cost = self.img_price * len(imgs_dict)
                assert (
                    len(imgs_dict) == n_image
                ), f"Unexpected number of image ({len(imgs_dict)} vs {n_image})"

                llm_dollar_cost = input_token_cost * self.llm_price["input_cost_per_token"]
                llm_dollar_cost += output_token_cost * self.llm_price["output_cost_per_token"]

                # add the image to anki collection
                imgs_name = self._update_anki_note(
                    note=note,
                    imgs_dict=imgs_dict,
                    reasonning=reasonning,
                    total_cost = img_cost + llm_dollar_cost,
                )
                whi("Image sent to anki.")

                # remove todo or failed tags
                removetags(nid=note["noteId"], tags="AnkiIllustrator::FAILED AnkiIllustrator::todo")

                # save to history
                with open(str(self.hisf / f"{time.time()}.json"), "w") as f:
                    new_hist = {
                        "note": note,
                        "imagenames": imgs_name,
                        "image_prompt": image_prompt,
                        "reasonning": reasonning,
                        "discarded text": discarded,
                        "llm_input_token_cost": input_token_cost,
                        "llm_output_token_cost": output_token_cost,
                        "llm_dollar_cost": llm_dollar_cost,
                        "llm_model": self.llm_model,
                        "Image_model": self.image_model,
                        "version": self.VERSION,
                        "date": self.today,
                        "trial": [d["trial"] for d in imgs_dict],
                    }
                    if self.image_model == "StableDiffusion":
                        new_hist["sd_steps"] = self.sd_steps
                        new_hist["StableDiffusion_dollar_cost"] = img_cost
                        new_hist["rseeds"] = [d["seed"] for d in imgs_dict]
                        new_hist["sd_cfg_scale"] = [d["cfg"]
                                                    for d in imgs_dict]
                    else:
                        new_hist["DALLE-E_dollar_cost"] = img_cost
                    json.dump(new_hist, f)
                    yel("History updated")

                if open_browser == "always":
                    whi("Openning anki browser")
                    anki(
                        action="guiBrowse",
                        query=f"nid:{note['noteId']}")

            except Exception as err:
                red(f"Exception in note loop: '{err}'")
                if debug:
                    raise
                else:
                    red("Adding tag 'illustrat:FAILED' to those cards")
                    addtags(nid=note["noteId"], tags="AnkiIllustrator::FAILED")

            # sync every few iterations
            if i % 100 == 0:
                sync_anki()

        # print total cost at the end
        self._total_cost()

        # add and remove the tag TODO to make it easier to readd by the user
        # as it was cleared by calling 'clearUnusedTags'
        nid = note['noteId']
        addtags(nid, tags="AnkiIllustrator::TODO")
        removetags(nid, tags="AnkiIllustrator::TODO")

        # sync at the end
        if do_sync:
            sync_anki()

        if open_browser:
            whi(f"Openning browser on query '{query}'")
            anki(action="guiBrowse", query=self.original_query)

        red("Finished.")
        return

    def _update_anki_note(
            self,
            note: Dict,
            imgs_dict: List[Dict],
            reasonning: str,
            total_cost: float,
        ):
        """
        Update an Anki note with generated images and metadata.

        Parameters
        ----------
        note : Dict
            Dictionary containing note information from Anki
        imgs_dict : List[Dict]
            List of dictionaries containing generated images and their metadata
        reasonning : str
            The reasoning behind the image generation
        total_cost : float
            Total cost of generating the images
        """
        nid = int(note["noteId"])
        whi(f"Editing anki note '{nid}'")

        def escape(input_str):
            """
            Escape HTML content and convert to plain text.
            
            Parameters
            ----------
            input_str : str
                Input string that may contain HTML
                
            Returns
            -------
            str
                Plain text with HTML removed and special characters escaped
            """
            soup = BeautifulSoup(input_str, "html.parser")
            if bool(soup.find()):
                input_str = soup.get_text()

            # remove quotation sign etc that would mess with the html
            input_str = input_str.replace('"', " ").replace(
                "<", " ").replace(">", " ")
            return input_str

        n = len(imgs_dict)
        contenthash = hashlib.md5(
            str(note["formatted_content"]).encode()).hexdigest()
        original_content = note["fields"]["AnkiIllustrator"]["value"].strip()
        full_html = ""
        imgs_name = []

        # put the images in a common container
        full_html += '<span class="AnkiIllustratorContainer">'

        for i, d in enumerate(imgs_dict):
            # creating unique filename
            img_hash = hashlib.md5(d["img_bin"].tobytes()).hexdigest()
            img_name = f"AnkiIllustrator_{contenthash[:25]}_{img_hash[:25]}.png"
            imgs_name.append(img_name)

            # saving image in history folder
            img_path = str((self.hisf / img_name).absolute())
            with open(img_path, "wb") as f:
                d["img_bin"].save(f)
                whi(f"Image {i+1}/{n} saved to '{img_path}'")

            # store the image and create the new field content
            result = anki(
                action="storeMediaFile",
                filename=img_name,
                path=img_path,
            )
            if result != img_name:
                raise Exception(f"output is {result} instead of {img_name}")
            else:
                # delete image from history file
                Path(img_path).unlink()

            # future field content
            field_content = f'<img src="{img_name}" '
            field_content += f'title="DATE:{self.today} '
            if "stablediffusion" in self.image_model.lower():
                field_content += f"STEPS:{self.sd_steps} "
            if n != 1:
                field_content += f"N:{i+1}/{n} "
            field_content += f"IMG_MODEL:{self.image_model} "
            field_content += f'TRIAL:{d["trial"]} '
            if self.image_model == "StableDiffusion":
                field_content += f'CFGSCALE:{d["cfg"]} '
                field_content += f'SEED:{d["seed"]} '
            field_content += f"LLMMODEL:{self.llm_model} "
            field_content += f"VERSION:{self.VERSION}\n"
            field_content += (
                f'CLOZE:{escape(note["formatted_content"])}\n'
            )
            field_content += f'REASONNING: {escape(reasonning)}\n\n'
            field_content += f'PROMPT: {escape(d["img_prompt"])}"'
            field_content += ' class="AnkiIllustratorImages"'
            field_content += ">"

            full_html += field_content

        # close the image container
        full_html += "</span><br>"

        # append reasonning just after the image(s)
        reason = escape(reasonning).splitlines()
        for i, li in enumerate(reason):
            sp = li.split(":", 1)
            if len(sp) == 2:
                reason[i] = f"<b>{sp[0].title()}</b>" + sp[1]
        reason = "<br>".join(reason)
        full_html += f"<br>{reason}"
        # and prompt
        full_html += f'<br><br><b>Prompt</b> "{escape(imgs_dict[0]["img_prompt"])}"'.replace("\n", "<br>")

        # add version and date
        full_html += f"<br>[DATE:{self.today} VERSION:{self.VERSION} LLMMODEL:{self.llm_model} IMAGEMODEL:{self.image_model} COST:{total_cost:.4f}]"

        # restore previous field content if nonempty
        if original_content:
            full_html += "<br><br>"

            # wrap the previous content in a detail tag
            # remove previous detail tag
            original_content = re.sub(
                r"\</?details\>|\</?summary\>", "", original_content
            )
            # also remove italics and other sentence just in case
            original_content = re.sub(
                r"\<i\>.*?\</i\>", "", original_content, flags=re.M | re.DOTALL
            )
            # keep only the images actually, to get rid of the span etc
            soup = BeautifulSoup(original_content, "html.parser")

            # remove possible duplication of old image
            passed_imgs = []
            for oldimg in soup.find_all("img"):
                if str(oldimg) not in passed_imgs:
                    passed_imgs.append(str(oldimg))

            # make sure the old images contain the right class
            for i, p in enumerate(passed_imgs):
                passed_imgs[i] = re.sub(
                    'class=".*?"',
                    'class="AnkiIllustratorOldImages"',
                    passed_imgs[i],
                )
                if "class" not in passed_imgs[i]:
                    assert passed_imgs[i][-1] == ">", "invalid passed img"
                    passed_imgs[i] = passed_imgs[i][:-1]
                    passed_imgs[i] += ' class="AnkiIllustratorOldImages">'

            # TODO keep only the last 3 generations based on date

            if (not passed_imgs) and "img" in original_content:
                # failed to parse passed only images
                breakpoint()

            original_content = "".join(passed_imgs)

            # remove previous style setting including size
            original_content = re.sub(
                r'style=".*?"',
                " ",
                original_content,
            )
            original_content = re.sub(
                r'((max-)?(width|height))[=:]"?\d+(px|%)?"?',
                " ",
                original_content,
            )
            full_html += "<!--SEPARATOR-->"
            full_html += "<details><summary>Previous illustrations</summary>"
            full_html += original_content
            full_html += "</details>"

        # makes sure to avoid having a close in the final field otherwise
        # "Empty cards..." will not work and you'll get an annoying
        # warning in the browser
        full_html = full_html.replace("}}", "]]")
        full_html = re.sub(r"\{\{c(\d+)::", r"[[c\1::", full_html)
        assert "{{c1::" not in full_html, f"Failed to substitute cloze markups before storing to field"

        # update the note
        yel("Updating note field")
        updatenote(nid, fields={"AnkiIllustrator": full_html})

        # add tag to updated note
        yel("Adding tag")
        addtags(nid=nid, tags=f"AnkiIllustrator::done::{self.today}")

        return imgs_name

    def _create_prompt(self, note_info: Dict):
        "create stable diffusion prompt from anki flashcard"
        # perfect llm stable diffusion prompt:
        # (subject of the image), (5 descriptive keyword), (camera type), (camera lens type), (time of day), (style of photography), (type of film), (Realism Level), (Best type of lighting for the subject)
        # source: https://www.youtube.com/watch?v=ZOTS3lShi-Y
        messages = semantic_prompt_filtering(
            curr_mess={"role": "user", "content": note_info["formatted_content"]},
            max_token=self.llm_max_token,
            temperature=0,
            prompt_messages=copy.deepcopy(self.dataset),
            keywords="",
            embedding_model=self.embedding_model,
            whi=whi,
            yel=yel,
            red=red,
        ) + [
                {
                    "role": "user",
                    "content": note_info["formatted_content"]
                    }
                ]

        # remove indentation of triple quotes
        for i, m in enumerate(messages):
            messages[i]["content"] = dedent(m["content"])

        input_token_cost = 0
        output_token_cost = 0

        assert tkn_len(messages) <= self.llm_max_token
        response = chat(
            messages=messages,
            model=self.llm_model,
            temperature=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            num_retries=5,
        )
        input_token_cost += response["usage"]["prompt_tokens"]
        output_token_cost += response["usage"]["completion_tokens"]
        reasonning, image_prompt, discarded_text = parse_llm_answer(response)
        whi(f"\n* Note formatted content: '{note_info['formatted_content']}'")
        whi(f"\n* Reasonning: '{reasonning}'")
        whi(f"\n* Prompt: '{image_prompt}'")
        whi(f"\n* Token cost: {input_token_cost} + {output_token_cost}")
        return input_token_cost, output_token_cost, image_prompt, reasonning, discarded_text

    def _extract_content(self):
        """
        Extract and format content from Anki notes.
        
        Processes note fields to:
        - Remove HTML and media
        - Apply custom formatting if configured
        - Add memory anchors if available
        - Add major system phonetic hints if enabled
        """
        for i, f in enumerate(tqdm(self.notes_info)):
            # deck = f["deckName"]
            fields = f["fields"]
            content = ""
            for fn in self.field_names:
                content += f"\n{fn.title()}: {fields[fn]['value'].strip()}"
            content = content.strip()

            orig_content = content

            content, media = replace_media(
                content=content,
                media=None,
                mode="remove_media")


            if self.string_formatting:
                content = self.cloze_input_parser(content)

            # identify anchors
            if self.anchors:
                matching_anchors = filter_anchors(
                    n=15,
                    content=content,
                    anchors=self.anchors,
                    embeddings=self.embeddings,
                    model=self.embedding_model,
                )

                anchors_to_add = " ; ".join([f"{k.strip()}: {v.strip()}" for k, v in matching_anchors]).strip()

                content += "\n\nAnchors: '" + anchors_to_add + "'"
                whi(f"Anchors: '{anchors_to_add.strip()}'")

            if self.major_system:
                # to make it easier to use the major system, help
                # the tokenizer by separating digits
                digit_prep = content
                # don't count numbers from anchors
                digit_prep = digit_prep.split("Anchors: ")[0].strip()
                assert digit_prep

                # remove cloze number if present
                digit_prep = re.sub(r"{{c\d+\s?(::)?|}}", "", digit_prep)
                for rom, rep in roman_numerals.items():
                    digit_prep = re.sub(r"(\W" + rom + "\W)", r"\1 (" + str(rep) + ")", digit_prep)

                # add space around digits
                digit_prep = re.sub(r"(\d)", r" \1 ", digit_prep)
                digit_prep = re.sub(r"(\d)\s+(\d)", r"\1 \2", digit_prep)
                digit_prep = re.sub(r"(\d)\s+(\d)", r"\1 \2", digit_prep)

                # find the digits, to help for major system
                # turn 370 into 37
                digit_prep = re.sub(r"(\d) 0", r"\1", digit_prep)
                # turn 30% into 3, and 0.5 into 5
                for symb in [".", ",", "%"]:
                    while f"0 {symb}" in digit_prep:
                        digit_prep = digit_prep.replace(f"0 {symb}", "")
                # find remaining digits
                temp = list(set([int(d) for d in re.findall(r"\d", digit_prep)]))
                if temp:
                    digits = []
                    [digits.append(d) for d in temp if d not in digits]
                    content += "\n\nPhonetic: "
                    for d in digits:
                        content += f"{d} as {major_table[d]} ; "
                    if content[-3:] == " ; ":
                        content = content[:-3]
                else:
                    content += "\n\nPhonetic: none"

            # strip html
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text()

            yel(f"\n\nOld content: '{orig_content}'")
            red(f"New content: '{content}'\n\n")
            print("")

            self.notes_info[i]["formatted_content"] = content

        # remove notes that are None, for example if containing an image
        self.notes_info = [passed for passed in self.notes_info if passed]

    def _load_stable_diffusion(self):
        """
        Initialize the Stable Diffusion API client.
        
        Returns
        -------
        StabilityInference
            Configured Stable Diffusion API client
        """
        whi("Loading stable api client")
        os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"

        assert "STABLEDIFFUSION_API_KEY" in os.environ, (
            "Missing stablediffusion api keys")
        stability_api = client.StabilityInference(
            key=self.sd_apikey,
            verbose=True,
            engine="stable-diffusion-512-v2-1",
            # engine="stable-diffusion-xl-beta-v2-2-2",  # more expensive one
        )
        whi("Loaded stable api client")

        return stability_api

    def _sanitize_image_prompt(self, prompt: str) -> [str, int, int]:
        """
        Sanitize image prompts to avoid content filter triggers.
        
        Uses LLM to rephrase prompts that might trigger content filters
        while preserving the core meaning.
        
        Parameters
        ----------
        prompt : str
            Original image generation prompt
            
        Returns
        -------
        Tuple[str, int, int]
            Sanitized prompt, input token cost, output token cost
        """
        messages = semantic_prompt_filtering(
            curr_mess={"role": "user", "content": prompt},
            max_token=self.llm_max_token,
            temperature=0,
            prompt_messages=copy.deepcopy(self.dataset_sanitize),
            keywords="",
            embedding_model=self.embedding_model,
            whi=whi,
            yel=yel,
            red=red,
            check_args={"must_be_unique": False}
        ) + [
                {
                    "role": "user",
                    "content": prompt
                    }
                ]

        san_input_token_cost = 0
        san_output_token_cost = 0
        assert tkn_len(messages) <= self.llm_max_token
        response = chat(
            messages=messages,
            model=self.llm_model,
            temperature=1,
            frequency_penalty=0,
            presence_penalty=0,
            num_retries=2,
        )
        san_input_token_cost += response["usage"]["prompt_tokens"]
        san_output_token_cost += response["usage"]["completion_tokens"]
        safe_prompt = response["choices"][0]["message"]["content"].strip(
        )
        ratio = levratio(prompt, safe_prompt)
        if ratio <= 85:
            raise Exception(
                red(
                    f"Safer prompt is not similar to the original prompt:\n* {prompt}\n* {safe_prompt}\n=> ratio: '{ratio}'"
                )
            )
        red(
            f"Succesfully sanitized prompt:\n* {prompt}\n* {safe_prompt}")
        return safe_prompt, san_input_token_cost, san_output_token_cost

    def _create_image_dalle(
        self, prompt, san_input_token_cost=0, san_output_token_cost=0, trial=0
    ):
        """
        Generate images using DALL-E API.
        
        Parameters
        ----------
        prompt : str
            Image generation prompt
        san_input_token_cost : int, default 0
            Token cost from previous sanitization attempts
        san_output_token_cost : int, default 0
            Token cost from previous sanitization attempts  
        trial : int, default 0
            Number of previous attempts at generating this image
            
        Returns
        -------
        List[Dict]
            List of dictionaries containing generated images and metadata
        """
        if trial > 0:
            sanitized = self._sanitize_image_prompt(prompt)
            prompt = sanitized[0]
            san_input_token_cost += sanitized[1]
            san_output_token_cost += sanitized[2]
        elif trial >= self.max_sanitize_trial:
            raise Exception(
                "Your request activated DALL-E API's safety filters and "
                "could not be processed."
                f"No more retries (trial #{trial})."
            )

        try:
            result = cached_image_generation(
                prompt=prompt,
                model="dall-e-3",
                quality="standard",
                size="1024x1024",
                style="natural",
                n=self.n_image,
                response_format="b64_json",
                num_retries=2,
                # seed=self.SEED,
            )
        except Exception as err:
            red(f"Error when creating image using DALL-E: {err}")
            return self._create_image_dalle(
                prompt=prompt,
                trial=trial + 1,
                san_input_token_cost=san_input_token_cost,
                san_output_token_cost=san_output_token_cost,
            )

        imgs = [r["b64_json"] for r in result["data"]]
        imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in imgs]

        if len(imgs) != self.n_image:
            red(f"Invalid number of images: {len(imgs)} vs {self.n_image}")

        out_dict = [
            {
                "img_bin": img,
                "img_prompt": prompt,
                "trial": trial,
                "san_input_token_cost": san_input_token_cost,
                "san_output_token_cost": san_output_token_cost,
            }
            for img in imgs
        ]
        return out_dict

    def _create_image_sd(self, prompt, seed, cfg_scale, trial=0):
        """
        stable diffusion pricing https://platform.stability.ai/docs/getting-started/credits-and-billing

        Parameters:
        -----------
        prompt: str
            prompt of the image
        seed: int
            seed to make image creation deterministic, is changed when retried
        cfg_scale, int
            how close the prompt influences the image
        trial: int
            starts at 0, is incremented every time there is a adult filter warning
            to avoid infinite loop

        Output:
        -------
        list of binary image
        """
        if trial == 1:
            prompt += ", funny, cartoon"
        elif trial == 2:
            prompt += ", friendly, uplifting"
        elif trial == 3:
            prompt += ", warm"
        elif trial > 3:
            raise Exception(
                "Your request activated StableDiffusion API's safety filters and "
                "could not be processed."
                f"No more retries (trial #{trial})."
            )

        # Set up our initial generation parameters.
        try:
            answers = self.stabapi.generate(
                prompt=prompt,
                seed=seed,
                steps=self.sd_steps,
                cfg_scale=cfg_scale,
                width=512,
                height=512,
                samples=1,
                sampler=generation.SAMPLER_K_DPMPP_2M,
            )
            answers = [a for a in answers]
        except Exception as err:
            red(f"Error when creating image: '{err}'")
            return self._create_image_sd(
                prompt=prompt,
                seed=seed * 2,
                trial=trial + 1,
                cfg_scale=cfg_scale,
            )

        # Set up our warning to print to the console if the adult content classifier is tripped.
        # If adult content classifier is tripped, retry with another seed until depth maxed out
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        f"Trial number #{trial}"
                    )
                    return self._create_image_sd(
                        prompt=prompt,
                        seed=seed * 2,
                        trial=trial + 1,
                        cfg_scale=cfg_scale,
                    )
                elif artifact.type == generation.ARTIFACT_IMAGE:
                    binary = artifact.binary
                    loaded = Image.open(io.BytesIO(binary))
                    return {
                        "img_bin": loaded,
                        "seed": seed,
                        "cfg": cfg_scale,
                        "trial": trial,
                        "img_prompt": prompt,  # store just in case it
                        # was modified by trials
                    }

    def _total_cost(self):
        """
        Calculate total cost of all image generations.
        
        Reads history JSON files to sum up costs from all previous runs.
        Also checks remaining Stable Diffusion credits if using that service.
        
        Returns
        -------
        float
            Total cost in dollars
        """
        json_files = [f for f in self.hisf.glob("*.json")]
        total_cost = 0
        for j in json_files:
            try:
                content = json.loads(Path(j).read_text())
                keys = [k for k in content.keys() if "dollar" in k.lower()]
                for k in keys:
                    val = float(content[k])
                    if val > 1:
                        val = val / 1000  # forgot to divide by 1000 the tkn cost at some point
                    total_cost += val
                    assert float(content[k]) > 0, "0 price?!"
            except Exception as err:
                red(f"Error when adding to total cost: '{err}' from file {j}")
        red(f"For the record the total cost of all calls in database is ${total_cost:01f}")

        # get balance in stablediffusion account
        if self.image_model == "StableDiffusion":
            response = requests.get(
                url="https://api.stability.ai/v1/user/balance",
                headers={"Authorization": f"Bearer {self.sd_apikey}"},
            )
            if response.status_code != 200:
                raise Exception(f"Non-200 response: {response.text}")
            balance_cred = float(response.json()["credits"])
            balance_dol = balance_cred * 10 / 1000
            red(
                f"Available StableDiffusion credits: {balance_cred:01f} (${balance_dol:01f})"
            )

            margin_image = int(balance_dol / self.img_price)
            margin_note = int(margin_image / self.n_image)
            red(f"This is enough for {margin_image} images ({margin_note} notes).")

        return total_cost


def parse_llm_answer(response : Dict) -> Tuple[str, str, str]:
    "simple test to check if llm response is correctly formatted"
    prp = response["choices"][0]["message"]["content"].strip()
    discarded = ""
    try:
        sp = prp.split("Answer: '")
        assert len(sp) == 2, f"Invalid LLM answer split length: {len(sp)}\nFull answer:\n'{prp}'"
        reasonning, prompt = sp[0].strip(), sp[1].strip()
        assert reasonning.strip(), f"Invalid LLM answer: empty reasonning: {reasonning}\nFull answer:\n'{prp}'"
        assert prompt.strip(), f"Invalid LLM answer: image prompt not found:\nFull answer:\n'{prp}'"
        while prompt[0] in [" ", "'", '"']:
            prompt = prompt[1:]
        while prompt[-1] in [" ", "'", '"']:
            prompt = prompt[:-1]
        assert prompt.strip(), f"Invalid LLM answer: image prompt not found:\nFull answer:\n'{prp}'"
        assert len(prompt.splitlines()) == 1, f"Invalid LLM answer: image prompt contained multiple lines:\nFull answer:\n'{prp}'"

    except Exception as err:
        red(f"Error when parsing LLM answer: '{err}'\nTrying another way.")
        lines = prp.splitlines(keepends=True)
        reasonning = []
        prompt = ""
        discarded_before = []
        discarded_after = []
        for li in lines:
            if li.startswith("Topic:"):
                reasonning.append(li)
                assert not discarded_after, f"Invalid LLM answer: found lines to discard before reasonning:\nFull answer:\n'{prp}'"
            elif not reasonning:
                discarded_before.append(li)
            elif prompt:
                discarded.append(li)
            elif li.startswith("Answer:"):
                prompt = li.replace("Answer:", "", 1).strip()
                assert reasonning, f"Invalid LLM answer: found image prompt before reasonning:\nFull answer:\n'{prp}'"
            else:
                reasonning.append(li)
        assert prompt, f"Invalid LLM answer: image prompt not found on 2nd parsing method:\nFull answer:\n'{prp}'"
        while prompt[0] in [" ", "'", '"']:
            prompt = prompt[1:]
        while prompt[-1] in [" ", "'", '"']:
            prompt = prompt[:-1]
        assert prompt, f"Invalid LLM answer: image prompt not found on 2nd parsing method:\nFull answer:\n'{prp}'"
        reasonning = "\n".join(reasonning).strip()
        discarded = "\n".join(discarded_before).strip() + "\n[REASONING+PROMPT]\n" + "\n".join(discarded_after).strip()
        if discarded:
            red(f"Found lines after the image prompt that are to be discarded instead of being included in the image prompt:\n'''\{discarded}\n'''")
        assert reasonning.strip(), f"Invalid llm empty reasonning: {reasonning} ({prp})"


    # extra check just in case
    assert len(prompt.splitlines()) == 1, f"Invalid LLM answer: image prompt contained multiple lines:\nFull answer:\n'{prp}'"
    assert reasonning.strip(), f"Invalid llm empty reasonning: {reasonning} ({prp})"

    prompt = re.sub("child(ren)?|kid", "young person", prompt.lower())

    return reasonning, prompt, discarded

def send_notif(content, url=None, disable=False):
    """
    Send a notification via ntfy.sh service.
    
    Parameters
    ----------
    content : str
        Notification message content
    url : str, optional
        ntfy.sh URL to send to
    disable : bool, default False
        If True, skip sending notification
    """
    if disable:
        return
    else:
        assert url
        assert content
    send_ntfy(
        url=url,
        title="AnkiIllustrator",
        content=content,
    )



def send_full_log_to_phone(url):
    """
    Send complete log file contents via notification.
    
    Parameters
    ----------
    url : str
        ntfy.sh URL to send notification to
    """
    log_content = Path(log_file).read_text()
    send_notif(content=f"FULL LOG: '{log_content}'", url=url)


if __name__ == "__main__":
    args, kwargs = fire.Fire(lambda *args, **kwargs: [args, kwargs])
    if "help" in kwargs:
        print(help(AnkiIllustrator))
        raise SystemExit()
    whi(f"Launching illustrator.py with args '{args}' and kwargs '{kwargs}'")

    try:
        asm = AnkiIllustrator(*args, **kwargs)
        red("Syncing")
        sync_anki()
        if "disable_notif" not in kwargs and "ntfy_url" in kwargs:
            send_full_log_to_phone(url=kwargs["ntfy_url"])

    except Exception as err:
        red(f"Exception: '{err}'")

        red("Syncing")
        sync_anki()

        if "debug" not in kwargs and "ntfy_url" in kwargs:
            send_notif(content=str(err), url=kwargs["ntfy_url"])
            red("Debugger not opened")
            if "disable_notif" not in kwargs and "ntfy_url" in kwargs:
                d = datetime.datetime.today()
                today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
                send_full_log_to_phone(url=kwargs["ntfy_url"])
        else:
            raise
