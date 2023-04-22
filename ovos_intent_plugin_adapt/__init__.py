from adapt.engine import IntentDeterminationEngine
from adapt.intent import IntentBuilder
from ovos_bus_client.session import SessionManager
from ovos_plugin_manager.intents import IntentExtractor, IntentPriority, IntentDeterminationStrategy, IntentMatch
from ovos_utils.log import LOG


class AdaptExtractor(IntentExtractor):
    def __init__(self, config=None,
                 strategy=IntentDeterminationStrategy.SEGMENT_REMAINDER,
                 priority=IntentPriority.KEYWORDS_HIGH,
                 segmenter=None):
        super().__init__(config, strategy=strategy, priority=priority, segmenter=segmenter)
        self.engines = {}  # lang: IntentDeterminationEngine
        self._excludes = {}  # workaround unmerged PR in adapt
        #  https://github.com/MycroftAI/adapt/pull/156

    def _get_engine(self, lang=None):
        lang = lang or self.lang
        if lang not in self.engines:
            self.engines[lang] = IntentDeterminationEngine()
        return self.engines[lang]

    def register_entity(self, entity_name, samples=None, lang=None):
        samples = samples or [entity_name]
        engine = self._get_engine(lang)
        for kw in samples:
            engine.register_entity(kw, entity_name)
        super().register_entity(entity_name, samples, lang)

    def register_regex_entity(self, entity_name, samples, lang=None):
        engine = self._get_engine(lang)
        if isinstance(samples, str):
            engine.register_regex_entity(samples)
        if isinstance(samples, list):
            for s in samples:
                engine.register_regex_entity(s)
        super().register_regex_entity(entity_name, samples, lang)

    def register_regex_intent(self, intent_name, samples, lang=None):
        self.register_regex_entity(intent_name, samples)
        self.register_keyword_intent(intent_name, [intent_name])
        super().register_regex_intent(intent_name, samples, lang)

    def register_keyword_intent(self, intent_name,
                                keywords=None,
                                optional=None,
                                at_least_one=None,
                                excluded=None,
                                lang=None):
        engine = self._get_engine(lang)
        if not keywords:
            keywords = keywords or [intent_name]
            self.register_entity(intent_name, keywords)
        optional = optional or []
        excluded = excluded or []
        at_least_one = at_least_one or []
        super().register_keyword_intent(intent_name, keywords, optional, at_least_one, excluded, lang)

        # structure intent
        intent = IntentBuilder(intent_name)
        for kw in keywords:
            intent.require(kw)
        for kw in optional:
            intent.optionally(kw)
        # TODO exclude functionality not yet merged
        #  https://github.com/MycroftAI/adapt/pull/156
        if excluded:
            self._excludes[intent_name] = excluded
        engine.register_intent_parser(intent.build())
        return intent

    def calc_intent(self, utterance, min_conf=0.5, lang=None, session=None):
        utterance = utterance.strip()
        engine = self._get_engine(lang)
        session = session or SessionManager.get()
        for intent in engine.determine_intent(utterance, 100,
                                              include_tags=True,
                                              context_manager=session.context):
            # workaround "excludes" keyword, just drop the intent match if we find an excluded keyword in utt
            if intent["intent_type"] in self._excludes:
                if any(w in utterance for w in self._excludes[intent["intent_type"]]):
                    return

            if intent and intent.get('confidence') >= min_conf:
                intent.pop("target")
                matches = {k: v for k, v in intent.items() if
                           k not in ["intent_type", "confidence", "__tags__"]}
                intent["entities"] = {}
                for k in matches:
                    intent["entities"][k] = intent.pop(k)
                intent["conf"] = intent.pop("confidence")
                intent["utterance"] = utterance
                intent["intent_engine"] = "adapt"

                remainder = self.get_utterance_remainder(utterance, samples=[v for v in matches.values()])
                intent["utterance_remainder"] = remainder
                skill_id = self.get_intent_skill_id(intent["intent_type"])
                return IntentMatch(intent_service=intent["intent_engine"],
                                   intent_type=intent["intent_type"],
                                   intent_data=intent,
                                   confidence=intent["conf"],
                                   skill_id=skill_id)
        return None

    def detach_intent(self, intent_name):
        super().detach_intent(intent_name)
        LOG.debug("detaching adapt intent: " + intent_name)
        for lang in self.engines:
            new_parsers = [p for p in self.engines[lang].intent_parsers
                           if p.name != intent_name]
            self.engines[lang].intent_parsers = new_parsers

    def detach_skill(self, skill_id):
        super().detach_skill(skill_id)
        LOG.debug("detaching adapt skill: " + skill_id)
        for lang in self.engines:
            new_parsers = [
                p.name for p in self.engines[lang].intent_parsers if
                p.name.startswith(skill_id)]
            for intent_name in new_parsers:
                self.detach_intent(intent_name)
