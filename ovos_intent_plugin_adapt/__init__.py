"""An intent parsing service using the Adapt parser."""
from threading import Lock

from adapt.engine import IntentDeterminationEngine
from ovos_bus_client.session import SessionManager
from ovos_plugin_manager.templates.pipeline import IntentPipelinePlugin, IntentMatch
from ovos_utils import flatten_list, classproperty
from ovos_utils.intents import IntentBuilder
from ovos_utils.log import LOG


def _munge(name, skill_id):  # avoid intent clashes between different skills using same names
    return f"{name}:{skill_id}"


def _unmunge(munged):
    return munged.split(":", 2)


class AdaptPipelinePlugin(IntentPipelinePlugin):
    """Intent service wrapping the Adapt intent Parser."""

    def __init__(self, bus, config=None):
        super().__init__(bus, config)
        # valid languages are known at boot time
        # TODO - use DomainIntentDeterminationEngine with langs as domains instead
        self.engines = {lang: IntentDeterminationEngine()
                        for lang in self.valid_languages}
        self.lock = Lock()
        self._excludes = {}  # workaround unmerged PR in adapt
        #  https://github.com/MycroftAI/adapt/pull/156

    # plugin api
    @classproperty
    def matcher_id(self):
        return "adapt"

    def match(self, utterances, lang, message):
        return self.match_intent(utterances, lang, message)

    # implementation
    def register_entity(self, skill_id, entity_name, samples, lang=None):
        lang = lang or self.lang  # magic property from message object
        super().register_keyword_intent(skill_id, entity_name, samples, lang)
        entity_name = _munge(entity_name, skill_id)
        if lang in self.engines:
            with self.lock:
                for s in samples:
                    self.engines[lang].register_entity(s, entity_name)

    def register_regex_entity(self, skill_id, entity_name, samples, lang=None):
        lang = lang or self.lang  # magic property from message object
        super().register_regex_entity(skill_id, entity_name, samples, lang)
        entity_name = _munge(entity_name, skill_id)  # TODO - munge regex_str
        if lang in self.engines:
            with self.lock:
                for s in samples:
                    self.engines[lang].register_regex_entity(s)

    def register_keyword_intent(self, skill_id, intent_name, required,
                                optional=None, at_least_one=None,
                                excluded=None, lang=None):
        lang = lang or self.lang  # magic property from message object
        super().register_keyword_intent(skill_id, intent_name, required,
                                        optional, at_least_one, excluded, lang)
        intent_name = _munge(intent_name, skill_id)
        intent = IntentBuilder(intent_name)
        for kw in required:
            intent.require(kw)
        for kw in optional:
            intent.optionally(kw)
        for kw in at_least_one:
            intent.at_least_one(kw)
        # NOTE - excluded not supported, PR not merged
        #  https://github.com/MycroftAI/adapt/pull/156
        if excluded:
            self._excludes[intent_name] = excluded  # HACK

        with self.lock:
            self.engines[lang].register_intent_parser(intent)

    def detach_skill(self, skill_id):
        """Remove all intents for skill.

        Args:
            skill_id (str): skill to process
        """
        with self.lock:
            for lang in self.engines:
                ents = []
                for entity in (e for e in self.registered_entities if e.skill_id == skill_id):
                    munged = _munge(entity.name, skill_id)
                    ents.append(munged)

                intents = []
                for intent in (e for e in self.registered_intents if e.skill_id == skill_id):
                    munged = _munge(intent.name, skill_id)
                    intents.append(munged)

                skill_parsers = [
                    p.name for p in self.engines[lang].intent_parsers if
                    p.name in munged  # munged skill_id/intent_name
                ]
                self.engines[lang].drop_intent_parser(skill_parsers)

                def match_skill_entities(data):
                    return data and data[1] in ents

                def match_skill_regexes(regexp):
                    return any([r in ents
                                for r in regexp.groupindex.keys()])

                self.engines[lang].drop_regex_entity(match_func=match_skill_regexes)
                self.engines[lang].drop_entity(match_func=match_skill_entities)

        super().detach_skill(skill_id)

    def detach_intent(self, skill_id, intent_name):
        super().detach_intent(skill_id, intent_name)
        intent_name = _munge(intent_name, skill_id)
        for lang in self.engines:
            new_parsers = [
                p for p in self.engines[lang].intent_parsers if p.name != intent_name
            ]
            self.engines[lang].intent_parsers = new_parsers

    def detach_entity(self, skill_id, entity_name):
        super().detach_entity(skill_id, entity_name)

        entity_name = _munge(entity_name, skill_id)

        def match_rx_enty(regexp):
            return any([r == entity_name
                        for r in regexp.groupindex.keys()])

        def match_skill_entities(data):
            return data and data[1] == entity_name

        for lang in self.engines:
            self.engines[lang].drop_entity(match_func=match_skill_entities)
            self.engines[lang].drop_regex_entity(match_func=match_rx_enty)

    def match_intent(self, utterances, lang=None, message=None):
        """Run the Adapt engine to search for a matching intent.

        Args:
            utterances (iterable): utterances for consideration in intent
            matching. As a practical matter, a single utterance will be
            passed in most cases.  But there are instances, such as
            streaming STT that could pass multiple.  Each utterance
            is represented as a tuple containing the raw, normalized, and
            possibly other variations of the utterance.

        Returns:
            Intent structure, or None if no match was found.
        """
        # we call flatten in case someone is sending the old style list of tuples
        utterances = flatten_list(utterances)
        lang = lang or self.lang
        if lang not in self.engines:
            return None

        best_intent = {}

        def take_best(intent, utt):
            nonlocal best_intent
            best = best_intent.get('confidence', 0.0) if best_intent else 0.0
            conf = intent.get('confidence', 0.0)
            if conf > best:
                best_intent = intent
                # TODO - Shouldn't Adapt do this?
                best_intent['utterance'] = utt

        sess = SessionManager.get(message)
        for utt in utterances:
            try:
                intents = [i for i in self.engines[lang].determine_intent(
                    utt, 100,
                    include_tags=True,
                    context_manager=sess.context)]

                # workaround "excludes" keyword, just drop the intent match if we find an excluded keyword in utt
                intents = [i for i in intents if not (
                        i["intent_type"] in self._excludes and
                        any(w in utt for w in self._excludes[i["intent_type"]])
                )]

                if intents:
                    utt_best = max(
                        intents, key=lambda x: x.get('confidence', 0.0)
                    )
                    take_best(utt_best, utt)

            except Exception as err:
                LOG.exception(err)

        if best_intent:
            ents = [tag['entities'][0] for tag in best_intent['__tags__'] if 'entities' in tag]

            sess.context.update_context(ents)

            skill_id = _unmunge(best_intent['intent_type'])[0]
            ret = IntentMatch(
                intent_service=self.matcher_id,
                intent_type=best_intent['intent_type'],
                intent_data=best_intent,
                skill_id=skill_id,
                utterance=best_intent['utterance'],
                confidence=best_intent["confidence"]
            )
        else:
            ret = None
        return ret
