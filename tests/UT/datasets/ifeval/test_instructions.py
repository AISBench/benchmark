"""Unit tests for ifeval instructions.py to increase coverage to 80%"""
import unittest
import random
from unittest.mock import patch, MagicMock

# Mock nltk.download 以避免在导入时阻塞
# instructions_util.py 在模块级别调用了 nltk.download('punkt_tab')，会导致阻塞
try:
    import nltk
    _original_download = nltk.download
    nltk.download = MagicMock(return_value=True)
except ImportError:
    pass

# 尝试导入指令模块
try:
    from ais_bench.benchmark.datasets.ifeval.instructions import (
        Instruction,
        ResponseLanguageChecker,
        NumberOfSentences,
        PlaceholderChecker,
        BulletListChecker,
        ConstrainedResponseChecker,
        ConstrainedStartChecker,
        HighlightSectionChecker,
        SectionChecker,
        ParagraphChecker,
        PostscriptChecker,
        RephraseChecker,
        KeywordChecker,
        KeywordFrequencyChecker,
        NumberOfWords,
        JsonFormat,
        ParagraphFirstWordCheck,
        KeySentenceChecker,
        ForbiddenWords,
        RephraseParagraph,
        TwoResponsesChecker,
        RepeatPromptThenAnswer,
        EndChecker,
        TitleChecker,
        LetterFrequencyChecker,
        CapitalLettersEnglishChecker,
        LowercaseLettersEnglishChecker,
        CommaChecker,
        CapitalWordFrequencyChecker,
        QuotationChecker,
    )
    INSTRUCTIONS_AVAILABLE = True
except ImportError:
    INSTRUCTIONS_AVAILABLE = False


class InstructionsTestBase(unittest.TestCase):
    """基础测试类"""
    @classmethod
    def setUpClass(cls):
        if not INSTRUCTIONS_AVAILABLE:
            cls.skipTest(cls, "Instructions modules not available")


class TestBaseInstruction(InstructionsTestBase):
    """测试基础 Instruction 类"""
    
    def test_instruction_init(self):
        """测试 Instruction 初始化"""
        inst = Instruction('test_id')
        self.assertEqual(inst.id, 'test_id')
    
    def test_build_description_not_implemented(self):
        """测试 build_description 未实现"""
        inst = Instruction('test_id')
        with self.assertRaises(NotImplementedError):
            inst.build_description()
    
    def test_get_instruction_args_not_implemented(self):
        """测试 get_instruction_args 未实现"""
        inst = Instruction('test_id')
        with self.assertRaises(NotImplementedError):
            inst.get_instruction_args()
    
    def test_get_instruction_args_keys_not_implemented(self):
        """测试 get_instruction_args_keys 未实现"""
        inst = Instruction('test_id')
        with self.assertRaises(NotImplementedError):
            inst.get_instruction_args_keys()
    
    def test_check_following_not_implemented(self):
        """测试 check_following 未实现"""
        inst = Instruction('test_id')
        with self.assertRaises(NotImplementedError):
            inst.check_following('test')


class TestResponseLanguageChecker(InstructionsTestBase):
    """测试 ResponseLanguageChecker"""
    
    def test_build_description_with_language(self):
        """测试指定语言构建描述"""
        checker = ResponseLanguageChecker('lang_check')
        desc = checker.build_description(language='en')
        self.assertIn('English', desc)
        self.assertEqual(checker._language, 'en')
    
    def test_build_description_random_language(self):
        """测试随机语言"""
        checker = ResponseLanguageChecker('lang_check')
        with patch('random.choice', return_value='fr'):
            desc = checker.build_description(language=None)
            self.assertIn('language', desc.lower())
    
    def test_get_instruction_args(self):
        """测试获取指令参数"""
        checker = ResponseLanguageChecker('lang_check')
        checker.build_description(language='en')  # 使用支持的语言代码
        args = checker.get_instruction_args()
        self.assertEqual(args['language'], 'en')
    
    def test_get_instruction_args_keys(self):
        """测试获取参数键"""
        checker = ResponseLanguageChecker('lang_check')
        keys = checker.get_instruction_args_keys()
        self.assertEqual(keys, ['language'])
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions.langdetect')
    def test_check_following_success(self, mock_langdetect):
        """测试语言检查成功"""
        mock_langdetect.detect.return_value = 'en'
        checker = ResponseLanguageChecker('lang_check')
        checker.build_description(language='en')
        result = checker.check_following('This is English text')
        self.assertTrue(result)
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions.langdetect')
    def test_check_following_exception(self, mock_langdetect):
        """测试语言检测异常"""
        mock_langdetect.detect.side_effect = Exception('Detection error')
        mock_langdetect.LangDetectException = Exception
        checker = ResponseLanguageChecker('lang_check')
        checker.build_description(language='en')
        result = checker.check_following('text')
        self.assertTrue(result)  # 异常时返回 True


class TestNumberOfSentences(InstructionsTestBase):
    """测试 NumberOfSentences"""
    
    def test_build_description_with_params(self):
        """测试指定参数构建描述"""
        checker = NumberOfSentences('num_sent')
        desc = checker.build_description(num_sentences=5, relation='at least')
        self.assertIn('at least', desc)
        self.assertIn('5', desc)
    
    def test_build_description_random_params(self):
        """测试随机参数"""
        checker = NumberOfSentences('num_sent')
        with patch('random.randint', return_value=3):
            with patch('random.choice', return_value='less than'):
                desc = checker.build_description(num_sentences=None, relation=None)
                self.assertIn('less than', desc)
    
    def test_build_description_invalid_relation(self):
        """测试无效关系"""
        checker = NumberOfSentences('num_sent')
        with self.assertRaises(ValueError):
            checker.build_description(num_sentences=5, relation='invalid')
    
    def test_get_instruction_args(self):
        """测试获取参数"""
        checker = NumberOfSentences('num_sent')
        checker.build_description(num_sentences=10, relation='at least')
        args = checker.get_instruction_args()
        self.assertEqual(args['num_sentences'], 10)
        self.assertEqual(args['relation'], 'at least')
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions_util._get_sentence_tokenizer')
    def test_check_following_less_than(self, mock_tokenizer):
        """测试少于阈值"""
        # Mock tokenizer 返回一个简单的 tokenizer，将文本按句号分割
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.tokenize = lambda text: [s for s in text.split('.') if s.strip()]
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        checker = NumberOfSentences('num_sent')
        checker.build_description(num_sentences=3, relation='less than')
        result = checker.check_following('One. Two.')
        self.assertTrue(result)
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions_util._get_sentence_tokenizer')
    def test_check_following_at_least(self, mock_tokenizer):
        """测试至少阈值"""
        # Mock tokenizer 返回一个简单的 tokenizer，将文本按句号分割
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.tokenize = lambda text: [s for s in text.split('.') if s.strip()]
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        checker = NumberOfSentences('num_sent')
        checker.build_description(num_sentences=2, relation='at least')
        result = checker.check_following('One. Two. Three.')
        self.assertTrue(result)


class TestPlaceholderChecker(InstructionsTestBase):
    """测试 PlaceholderChecker"""
    
    def test_build_description_with_placeholders(self):
        """测试指定占位符"""
        checker = PlaceholderChecker('placeholder')
        desc = checker.build_description(num_placeholders=2)
        self.assertIn('placeholder', desc.lower())
    
    def test_build_description_random(self):
        """测试随机占位符数量"""
        checker = PlaceholderChecker('placeholder')
        with patch('random.randint', return_value=3):
            desc = checker.build_description(num_placeholders=None)
            self.assertIsNotNone(desc)
    
    def test_check_following_success(self):
        """测试占位符检查成功"""
        checker = PlaceholderChecker('placeholder')
        checker.build_description(num_placeholders=2)
        result = checker.check_following('Text with [placeholder 1] and [placeholder 2]')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试占位符检查失败"""
        checker = PlaceholderChecker('placeholder')
        checker.build_description(num_placeholders=2)
        result = checker.check_following('Text without placeholders')
        self.assertFalse(result)


class TestBulletListChecker(InstructionsTestBase):
    """测试 BulletListChecker"""
    
    def test_build_description_with_num_bullets(self):
        """测试指定项目符号数量"""
        checker = BulletListChecker('bullet')
        desc = checker.build_description(num_bullets=3)
        self.assertIn('bullet', desc.lower())
    
    def test_check_following_success(self):
        """测试项目符号检查成功"""
        checker = BulletListChecker('bullet')
        checker.build_description(num_bullets=2)
        result = checker.check_following('* Item 1\n* Item 2')
        self.assertTrue(result)


class TestConstrainedResponseChecker(InstructionsTestBase):
    """测试 ConstrainedResponseChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = ConstrainedResponseChecker('constrained')
        desc = checker.build_description()
        self.assertIn('answer', desc.lower())
    
    def test_check_following_yes(self):
        """测试检查 yes 响应"""
        checker = ConstrainedResponseChecker('constrained')
        checker.build_description()
        result = checker.check_following('My answer is yes.')
        self.assertTrue(result)
    
    def test_check_following_no(self):
        """测试检查 no 响应"""
        checker = ConstrainedResponseChecker('constrained')
        checker.build_description()
        result = checker.check_following('My answer is no.')
        self.assertTrue(result)
    
    def test_check_following_invalid(self):
        """测试无效响应"""
        checker = ConstrainedResponseChecker('constrained')
        checker.build_description()
        result = checker.check_following('Some other response')
        self.assertFalse(result)


class TestConstrainedStartChecker(InstructionsTestBase):
    """测试 ConstrainedStartChecker"""
    
    def test_build_description_with_starter(self):
        """测试指定起始词"""
        checker = ConstrainedStartChecker('start')
        desc = checker.build_description(starter='I believe')
        self.assertIn('I believe', desc)
    
    def test_build_description_random(self):
        """测试随机起始词"""
        checker = ConstrainedStartChecker('start')
        with patch('random.choice', return_value='I think'):
            desc = checker.build_description(starter=None)
            self.assertIn('I think', desc)
    
    def test_check_following_success(self):
        """测试起始词检查成功"""
        checker = ConstrainedStartChecker('start')
        checker.build_description(starter='I believe')
        result = checker.check_following('I believe this is correct.')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试起始词检查失败"""
        checker = ConstrainedStartChecker('start')
        checker.build_description(starter='I believe')
        result = checker.check_following('This is my answer.')
        self.assertFalse(result)


class TestHighlightSectionChecker(InstructionsTestBase):
    """测试 HighlightSectionChecker"""
    
    def test_build_description_with_num_highlights(self):
        """测试指定高亮数量"""
        checker = HighlightSectionChecker('highlight')
        desc = checker.build_description(num_highlights=2)
        self.assertIn('highlight', desc.lower())
    
    def test_check_following_success(self):
        """测试高亮检查成功"""
        checker = HighlightSectionChecker('highlight')
        checker.build_description(num_highlights=2)
        result = checker.check_following('Text with *highlighted 1* and *highlighted 2*')
        self.assertTrue(result)


class TestSectionChecker(InstructionsTestBase):
    """测试 SectionChecker"""
    
    def test_build_description_with_section_splitter(self):
        """测试指定分节符"""
        checker = SectionChecker('section')
        # 参数名是 section_spliter 不是 section_splitter
        desc = checker.build_description(section_spliter='Section', num_sections=3)
        self.assertIn('Section', desc)
    
    def test_check_following_success(self):
        """测试分节检查成功"""
        checker = SectionChecker('section')
        checker.build_description(section_spliter='Section', num_sections=2)
        result = checker.check_following('Section 1\nContent\nSection 2\nContent')
        self.assertTrue(result)


class TestParagraphChecker(InstructionsTestBase):
    """测试 ParagraphChecker"""
    
    def test_build_description_with_num_paragraphs(self):
        """测试指定段落数量"""
        checker = ParagraphChecker('paragraph')
        desc = checker.build_description(num_paragraphs=3)
        self.assertIn('paragraph', desc.lower())
    
    def test_check_following_success(self):
        """测试段落检查成功"""
        checker = ParagraphChecker('paragraph')
        checker.build_description(num_paragraphs=2)
        # 段落必须用 *** 分隔
        result = checker.check_following('Paragraph 1.\n***\nParagraph 2.')
        self.assertTrue(result)


class TestPostscriptChecker(InstructionsTestBase):
    """测试 PostscriptChecker"""
    
    def test_build_description_with_postscript_marker(self):
        """测试指定附言标记"""
        checker = PostscriptChecker('postscript')
        desc = checker.build_description(postscript_marker='P.S.')
        self.assertIn('P.S.', desc)
    
    def test_check_following_success(self):
        """测试附言检查成功"""
        checker = PostscriptChecker('postscript')
        checker.build_description(postscript_marker='P.S.')
        result = checker.check_following('Main text.\nP.S. Additional note.')
        self.assertTrue(result)


class TestRephraseChecker(InstructionsTestBase):
    """测试 RephraseChecker"""
    
    def test_build_description_with_original_message(self):
        """测试指定原始消息"""
        checker = RephraseChecker('rephrase')
        # 原始消息必须包含 *change me* 格式
        desc = checker.build_description(original_message='Test *change me* message')
        self.assertIn('rephrase', desc.lower())
    
    def test_check_following(self):
        """测试改写检查"""
        checker = RephraseChecker('rephrase')
        checker.build_description(original_message='Original *text*')
        # 响应也必须包含 *change me* 格式
        result = checker.check_following('Original *different*')
        self.assertTrue(result)


class TestKeywordChecker(InstructionsTestBase):
    """测试 KeywordChecker"""
    
    def test_build_description_with_keywords(self):
        """测试指定关键词"""
        checker = KeywordChecker('keyword')
        desc = checker.build_description(keywords=['test', 'keyword'])
        self.assertIn('keyword', desc.lower())
    
    def test_check_following_success(self):
        """测试关键词检查成功"""
        checker = KeywordChecker('keyword')
        checker.build_description(keywords=['test', 'keyword'])
        result = checker.check_following('This is a test with keyword included.')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试关键词检查失败"""
        checker = KeywordChecker('keyword')
        checker.build_description(keywords=['test', 'keyword'])
        result = checker.check_following('This text is missing required words.')
        self.assertFalse(result)


class TestKeywordFrequencyChecker(InstructionsTestBase):
    """测试 KeywordFrequencyChecker"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = KeywordFrequencyChecker('freq')
        desc = checker.build_description(keyword='test', frequency=3, relation='at least')
        self.assertIn('test', desc)
    
    def test_check_following_success(self):
        """测试频率检查成功"""
        checker = KeywordFrequencyChecker('freq')
        checker.build_description(keyword='test', frequency=2, relation='at least')
        result = checker.check_following('test test test')
        self.assertTrue(result)


class TestNumberOfWords(InstructionsTestBase):
    """测试 NumberOfWords"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = NumberOfWords('words')
        desc = checker.build_description(num_words=100, relation='at least')
        self.assertIn('100', desc)
    
    def test_check_following_at_least(self):
        """测试至少字数"""
        checker = NumberOfWords('words')
        checker.build_description(num_words=5, relation='at least')
        result = checker.check_following('one two three four five six')
        self.assertTrue(result)
    
    def test_check_following_less_than(self):
        """测试少于字数"""
        checker = NumberOfWords('words')
        checker.build_description(num_words=10, relation='less than')
        result = checker.check_following('one two three')
        self.assertTrue(result)


class TestJsonFormat(InstructionsTestBase):
    """测试 JsonFormat"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = JsonFormat('json')
        desc = checker.build_description()
        self.assertIn('json', desc.lower())
    
    def test_check_following_valid_json(self):
        """测试有效 JSON"""
        checker = JsonFormat('json')
        checker.build_description()
        result = checker.check_following('{"key": "value"}')
        self.assertTrue(result)
    
    def test_check_following_invalid_json(self):
        """测试无效 JSON"""
        checker = JsonFormat('json')
        checker.build_description()
        result = checker.check_following('not json')
        self.assertFalse(result)


class TestParagraphFirstWordCheck(InstructionsTestBase):
    """测试 ParagraphFirstWordCheck"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = ParagraphFirstWordCheck('first_word')
        desc = checker.build_description(num_paragraphs=2, nth_paragraph=1, first_word='Test')
        self.assertIn('paragraph', desc.lower())
    
    def test_check_following_success(self):
        """测试首词检查成功"""
        checker = ParagraphFirstWordCheck('first_word')
        checker.build_description(num_paragraphs=2, nth_paragraph=1, first_word='Hello')
        result = checker.check_following('Hello world.\n\nSecond paragraph.')
        self.assertTrue(result)


class TestKeySentenceChecker(InstructionsTestBase):
    """测试 KeySentenceChecker"""
    
    def test_build_description_with_key_sentences(self):
        """测试指定关键句"""
        checker = KeySentenceChecker('key_sent')
        desc = checker.build_description(key_sentences=['Sentence one', 'Sentence two'], num_sentences=2)
        self.assertIn('sentence', desc.lower())
    
    def test_check_following_success(self):
        """测试关键句检查成功"""
        checker = KeySentenceChecker('key_sent')
        # 关键句必须是完整的句子，不能是句子的一部分
        checker.build_description(key_sentences=['Key sentence.'], num_sentences=1)
        result = checker.check_following('This is text. Key sentence. More text.')
        self.assertTrue(result)


class TestForbiddenWords(InstructionsTestBase):
    """测试 ForbiddenWords"""
    
    def test_build_description_with_forbidden_words(self):
        """测试指定禁用词"""
        checker = ForbiddenWords('forbidden')
        desc = checker.build_description(forbidden_words=['bad', 'wrong'])
        self.assertIn('not', desc.lower())
    
    def test_check_following_success(self):
        """测试禁用词检查成功"""
        checker = ForbiddenWords('forbidden')
        checker.build_description(forbidden_words=['bad', 'wrong'])
        result = checker.check_following('This is good text.')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试禁用词检查失败"""
        checker = ForbiddenWords('forbidden')
        checker.build_description(forbidden_words=['bad', 'wrong'])
        result = checker.check_following('This is bad text.')
        self.assertFalse(result)


class TestRephraseParagraph(InstructionsTestBase):
    """测试 RephraseParagraph"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = RephraseParagraph('repara')
        desc = checker.build_description(original_paragraph='Original text', low=5, high=10)
        self.assertIn('rephrase', desc.lower())
    
    def test_check_following(self):
        """测试改写段落检查"""
        checker = RephraseParagraph('repara')
        # 原始段落有5个词，响应需要有3-10个相同的词
        checker.build_description(original_paragraph='This is a test paragraph', low=1, high=3)
        # 响应有2个相同的词 (is, a)
        result = checker.check_following('This is a different text')
        self.assertTrue(result)


class TestTwoResponsesChecker(InstructionsTestBase):
    """测试 TwoResponsesChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = TwoResponsesChecker('two_resp')
        desc = checker.build_description()
        self.assertIn('two', desc.lower())
    
    def test_check_following_success(self):
        """测试两个响应检查成功"""
        checker = TwoResponsesChecker('two_resp')
        checker.build_description()
        result = checker.check_following('Response 1\n\n******\n\nResponse 2')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试两个响应检查失败"""
        checker = TwoResponsesChecker('two_resp')
        checker.build_description()
        result = checker.check_following('Only one response')
        self.assertFalse(result)


class TestRepeatPromptThenAnswer(InstructionsTestBase):
    """测试 RepeatPromptThenAnswer"""
    
    def test_build_description_with_prompt(self):
        """测试指定提示"""
        checker = RepeatPromptThenAnswer('repeat')
        desc = checker.build_description(prompt_to_repeat='Test prompt')
        self.assertIn('repeat', desc.lower())
    
    def test_check_following_success(self):
        """测试重复提示检查成功"""
        checker = RepeatPromptThenAnswer('repeat')
        checker.build_description(prompt_to_repeat='Test prompt')
        result = checker.check_following('Test prompt\n\nMy answer here.')
        self.assertTrue(result)


class TestEndChecker(InstructionsTestBase):
    """测试 EndChecker"""
    
    def test_build_description_with_end_phrase(self):
        """测试指定结束短语"""
        checker = EndChecker('end')
        desc = checker.build_description(end_phrase='The end.')
        self.assertIn('end', desc.lower())
    
    def test_check_following_success(self):
        """测试结束短语检查成功"""
        checker = EndChecker('end')
        checker.build_description(end_phrase='The end.')
        result = checker.check_following('Some text here. The end.')
        self.assertTrue(result)


class TestTitleChecker(InstructionsTestBase):
    """测试 TitleChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = TitleChecker('title')
        desc = checker.build_description()
        self.assertIn('title', desc.lower())
    
    def test_check_following_success(self):
        """测试标题检查成功"""
        checker = TitleChecker('title')
        checker.build_description()
        result = checker.check_following('<<Title Here>>\nContent follows.')
        self.assertTrue(result)
    
    def test_check_following_failure(self):
        """测试标题检查失败"""
        checker = TitleChecker('title')
        checker.build_description()
        result = checker.check_following('No title in this text.')
        self.assertFalse(result)


class TestLetterFrequencyChecker(InstructionsTestBase):
    """测试 LetterFrequencyChecker"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = LetterFrequencyChecker('letter_freq')
        desc = checker.build_description(letter='a', let_frequency=5, let_relation='at least')
        self.assertIn('letter', desc.lower())
    
    def test_check_following_success(self):
        """测试字母频率检查成功"""
        checker = LetterFrequencyChecker('letter_freq')
        checker.build_description(letter='a', let_frequency=3, let_relation='at least')
        result = checker.check_following('aaa bbb ccc')
        self.assertTrue(result)


class TestCapitalLettersEnglishChecker(InstructionsTestBase):
    """测试 CapitalLettersEnglishChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = CapitalLettersEnglishChecker('capital')
        desc = checker.build_description()
        self.assertIn('capital', desc.lower())
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions.langdetect')
    def test_check_following_all_caps(self, mock_langdetect):
        """测试全大写"""
        mock_langdetect.detect.return_value = 'en'
        mock_langdetect.LangDetectException = Exception
        
        checker = CapitalLettersEnglishChecker('capital')
        checker.build_description()
        result = checker.check_following('THIS IS ALL CAPS')
        self.assertTrue(result)
    
    def test_check_following_not_all_caps(self):
        """测试非全大写"""
        checker = CapitalLettersEnglishChecker('capital')
        checker.build_description()
        result = checker.check_following('This is not all caps')
        self.assertFalse(result)


class TestLowercaseLettersEnglishChecker(InstructionsTestBase):
    """测试 LowercaseLettersEnglishChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = LowercaseLettersEnglishChecker('lowercase')
        desc = checker.build_description()
        self.assertIn('lowercase', desc.lower())
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions.langdetect')
    def test_check_following_all_lower(self, mock_langdetect):
        """测试全小写"""
        mock_langdetect.detect.return_value = 'en'
        mock_langdetect.LangDetectException = Exception
        
        checker = LowercaseLettersEnglishChecker('lowercase')
        checker.build_description()
        result = checker.check_following('this is all lowercase')
        self.assertTrue(result)


class TestCommaChecker(InstructionsTestBase):
    """测试 CommaChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = CommaChecker('comma')
        desc = checker.build_description()
        self.assertIn('comma', desc.lower())
    
    def test_check_following_no_comma(self):
        """测试无逗号"""
        checker = CommaChecker('comma')
        checker.build_description()
        result = checker.check_following('Text without comma')
        self.assertTrue(result)
    
    def test_check_following_with_comma(self):
        """测试有逗号"""
        checker = CommaChecker('comma')
        checker.build_description()
        result = checker.check_following('Text with, comma')
        self.assertFalse(result)


class TestCapitalWordFrequencyChecker(InstructionsTestBase):
    """测试 CapitalWordFrequencyChecker"""
    
    def test_build_description_with_params(self):
        """测试指定参数"""
        checker = CapitalWordFrequencyChecker('cap_word_freq')
        desc = checker.build_description(capital_frequency=5, capital_relation='at least')
        self.assertIn('capital', desc.lower())
    
    @patch('ais_bench.benchmark.datasets.ifeval.instructions.instructions_util.nltk.word_tokenize')
    def test_check_following_success(self, mock_word_tokenize):
        """测试大写词频率检查成功"""
        # Mock word_tokenize 返回简单的单词列表
        mock_word_tokenize.return_value = ['WORD', 'ONE', 'and', 'WORD', 'TWO', 'here']
        
        checker = CapitalWordFrequencyChecker('cap_word_freq')
        checker.build_description(capital_frequency=2, capital_relation='at least')
        result = checker.check_following('WORD ONE and WORD TWO here')
        self.assertTrue(result)


class TestQuotationChecker(InstructionsTestBase):
    """测试 QuotationChecker"""
    
    def test_build_description(self):
        """测试构建描述"""
        checker = QuotationChecker('quote')
        desc = checker.build_description()
        self.assertIn('wrap', desc.lower())
    
    def test_check_following_double_quotes(self):
        """测试双引号"""
        checker = QuotationChecker('quote')
        checker.build_description()
        result = checker.check_following('"This is quoted text"')
        self.assertTrue(result)
    
    def test_check_following_no_quotes(self):
        """测试无引号"""
        checker = QuotationChecker('quote')
        checker.build_description()
        result = checker.check_following('This is not quoted')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

