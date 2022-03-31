from adapt.engine import IntentDeterminationEngine
from adapt.intent import IntentBuilder
from ovos_utils.log import LOG
from ovos_plugin_manager.intents import IntentExtractor, IntentPriority, IntentDeterminationStrategy


class AdaptExtractor(IntentExtractor):
    keyword_based = True
    regex_entity_support = True

    def __init__(self, config=None,
                 strategy=IntentDeterminationStrategy.SEGMENT_REMAINDER,
                 priority=IntentPriority.KEYWORDS_HIGH,
                 segmenter=None):
        super().__init__(config, strategy=strategy,
                         priority=priority, segmenter=segmenter)
        self.engine = IntentDeterminationEngine()

    def register_entity(self, entity_name, samples=None):
        samples = samples or [entity_name]
        for kw in samples:
            self.engine.register_entity(kw, entity_name)
        super().register_entity(entity_name, samples)

    def register_regex_entity(self, entity_name, samples):
        if isinstance(samples, str):
            self.engine.register_regex_entity(samples)
        if isinstance(samples, list):
            for s in samples:
                self.engine.register_regex_entity(s)

    def register_regex_intent(self, intent_name, samples):
        self.register_regex_entity(intent_name + "_adapt_rx", samples)
        self.register_intent(intent_name, [intent_name + "_adapt_rx"])

    def register_intent(self, intent_name, samples=None,
                        optional_samples=None, rx_samples=None):
        """
        :param intent_name: intent_name
        :param samples: list of required registered entities (names)
        :param optional_samples: list of optional registered samples (names)
        :return:
        """
        super().register_intent(intent_name, samples)
        if not samples:
            samples = [intent_name]
            self.register_entity(intent_name, samples)
        optional_samples = optional_samples or []

        # structure intent
        intent = IntentBuilder(intent_name)
        for kw in samples:
            intent.require(kw)
        for kw in optional_samples:
            intent.optionally(kw)
        self.engine.register_intent_parser(intent.build())
        return intent

    def calc_intent(self, utterance, min_conf=0.0):
        utterance = utterance.strip()
        if self.normalize:
            utterance = self.normalize_utterance(utterance, self.lang, True)
        for intent in self.engine.determine_intent(utterance, 100,
                                                   include_tags=True,
                                                   context_manager=self.context_manager):
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

                remainder = self.get_utterance_remainder(
                    utterance, samples=[v for v in matches.values()])
                intent["utterance_remainder"] = remainder
                return intent
        return {"conf": 0, "intent_type": "unknown", "entities": {},
                "utterance_remainder": utterance,
                "utterance": utterance, "intent_engine": "adapt"}

    def detach_intent(self, intent_name):
        super().detach_intent(intent_name)
        LOG.debug("detaching adapt intent: " + intent_name)
        new_parsers = [
            p for p in self.engine.intent_parsers if p.name != intent_name]
        self.engine.intent_parsers = new_parsers

    def detach_skill(self, skill_id):
        super().detach_skill(skill_id)
        LOG.debug("detaching adapt skill: " + skill_id)
        new_parsers = [
            p.name for p in self.engine.intent_parsers if
            p.name.startswith(skill_id)]
        for intent_name in new_parsers:
            self.detach_intent(intent_name)

    def manifest(self):
        # TODO vocab, skill ids, intent_data
        return {
            "intent_names": [p.name for p in self.engine.intent_parsers]
        }
