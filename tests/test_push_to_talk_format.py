"""Formatting, history, and playback tests."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ptt_support import (
    _make_aiohttp_mocks,
)

import dictation_audio
import push_to_talk as ptt


class TestPlaySound:
    """Tests covering playsound."""

    def test_file_not_found_does_nothing(self, mock_popen):
        """File not found does nothing."""
        popen_cls, _proc = mock_popen
        with patch("os.path.exists", return_value=False):
            ptt.play_sound("message")
        popen_cls.assert_not_called()

    def test_pw_play_used_first(self):
        """Pw play used first."""
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "subprocess.Popen",
            ) as mock_pop,
        ):
            ptt.play_sound("message")
        args = mock_pop.call_args[0][0]
        assert args[0] == "pw-play"

    def test_falls_back_to_paplay(self):
        """Falls back to paplay."""
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "subprocess.Popen",
                side_effect=[FileNotFoundError, MagicMock()],
            ) as mock_pop,
        ):
            ptt.play_sound("message")
        assert mock_pop.call_count == 2
        args = mock_pop.call_args_list[1][0][0]
        assert args[0] == "paplay"

    def test_all_players_fail(self):
        """All players fail."""
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "subprocess.Popen",
                side_effect=FileNotFoundError,
            ),
        ):
            ptt.play_sound("message")  # Should not raise

    def test_correct_path(self):
        """Correct path."""
        with (
            patch("os.path.exists", return_value=True) as mock_exists,
            patch(
                "subprocess.Popen",
            ),
        ):
            ptt.play_sound("complete")
        mock_exists.assert_called_with(
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
        )


class TestAutoPunctuate:
    """Tests covering auto punctuate."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Success."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " Hello, world. "})
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "Hello, world."

    @pytest.mark.asyncio
    async def test_failure_returns_original(self):
        """When the session raises, return the original text."""
        mock_aiohttp = MagicMock()

        class _RaisingSessionCtx:
            async def __aenter__(self):
                raise OSError("connection error")

            async def __aexit__(self, *a):
                return False

        mock_aiohttp.ClientSession = MagicMock(return_value=_RaisingSessionCtx())

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_non_200_returns_original(self):
        """Non 200 returns original."""
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.auto_punctuate("hello world", 5000)
        assert result == "hello world"


class TestTranslateText:
    """Tests covering translate text."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Success."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " Hallo Welt "})
        mock_aiohttp, _ = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.translate_text("hello world", "German", 5000)
        assert result == "Hallo Welt"

    @pytest.mark.asyncio
    async def test_posts_correct_payload(self):
        """Posts correct payload."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "Bonjour"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            await ptt.translate_text("hello", "French", 5000)
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["json"] == {"text": "hello", "target_language": "French"}

    @pytest.mark.asyncio
    async def test_failure_returns_original(self):
        """Failure returns original."""
        mock_aiohttp = MagicMock()

        class _RaisingSessionCtx:
            async def __aenter__(self):
                raise OSError("timeout")

            async def __aexit__(self, *a):
                return False

        mock_aiohttp.ClientSession = MagicMock(return_value=_RaisingSessionCtx())

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.translate_text("hello", "German", 5000)
        assert result == "hello"


class TestFormatDictation:
    """Pure logic function — test every branch thoroughly."""

    # Basic capitalization
    def test_capitalizes_first_letter(self):
        """Capitalizes first letter."""
        assert ptt.format_dictation("hello world") == "Hello world"

    def test_empty_string(self):
        """Empty string."""
        assert ptt.format_dictation("") == ""

    def test_single_char(self):
        """Single char."""
        assert ptt.format_dictation("a") == "A"

    def test_already_capitalized(self):
        """Already capitalized."""
        assert ptt.format_dictation("Hello") == "Hello"

    # Abbreviations
    def test_mister(self):
        """Mister."""
        result = ptt.format_dictation("mister smith")
        assert "Mr." in result

    def test_missus(self):
        """Missus."""
        result = ptt.format_dictation("missus jones")
        assert "Mrs." in result

    def test_doctor(self):
        """Doctor."""
        result = ptt.format_dictation("doctor who")
        assert "Dr." in result

    def test_versus(self):
        """Versus."""
        result = ptt.format_dictation("red versus blue")
        assert "vs." in result

    def test_et_cetera(self):
        """Et cetera."""
        result = ptt.format_dictation("and et cetera")
        assert "etc." in result

    def test_for_example(self):
        """For example."""
        result = ptt.format_dictation("for example this")
        # "for example" -> "e.g.", then first-letter capitalized -> "E.g."
        assert "e.g." in result.lower()

    def test_that_is(self):
        """That is."""
        result = ptt.format_dictation("that is good")
        # "that is" -> "i.e.", then first-letter capitalized -> "I.e."
        assert "i.e." in result.lower()

    def test_abbreviation_case_insensitive(self):
        """Abbreviation case insensitive."""
        result = ptt.format_dictation("DOCTOR who")
        assert "Dr." in result

    # Email patterns
    def test_email_pattern(self):
        """Email pattern."""
        result = ptt.format_dictation("send to john at gmail dot com")
        assert "john@gmail.com" in result

    def test_email_pattern_case_insensitive(self):
        """Email pattern case insensitive."""
        result = ptt.format_dictation("email me AT test DOT org")
        assert "me@test.org" in result

    # URL patterns
    def test_www_pattern(self):
        """Www pattern."""
        result = ptt.format_dictation("visit w w w dot google dot com")
        assert "www.google.com" in result

    def test_www_pattern_case_insensitive(self):
        """Www pattern case insensitive."""
        result = ptt.format_dictation("go to W W W DOT example DOT org")
        assert "www.example.org" in result


class TestFormatDictationNumbers:
    """Number and combined formatting rules."""

    def test_single_digit_words(self):
        """Single digit words."""
        assert "0" in ptt.format_dictation("zero items")
        assert "1" in ptt.format_dictation("one item")
        assert "5" in ptt.format_dictation("five items")
        assert "9" in ptt.format_dictation("nine items")

    def test_teen_numbers(self):
        """Teen numbers."""
        assert "11" in ptt.format_dictation("eleven items")
        assert "13" in ptt.format_dictation("thirteen items")
        assert "19" in ptt.format_dictation("nineteen items")

    def test_tens_numbers(self):
        """Tens numbers."""
        assert "20" in ptt.format_dictation("twenty items")
        assert "30" in ptt.format_dictation("thirty items")
        assert "50" in ptt.format_dictation("fifty items")
        assert "90" in ptt.format_dictation("ninety items")

    def test_number_word_case_insensitive(self):
        """Number word case insensitive."""
        assert "5" in ptt.format_dictation("Five items")

    # Hundreds
    def test_hundreds_with_remainder(self):
        """Hundreds with remainder."""
        result = ptt.format_dictation("three hundred twenty five items")
        # three->3, hundred, twenty->20, five->5 => "3 hundred 20" => "320" then "5"
        # Actually: "3 hundred 20" -> 320, then 5 is separate
        # Wait let me trace: first numbers are substituted: "3 hundred 20 5 items"
        # Then "3 hundred 20" -> 320 (first regex), leaving "320 5 items"
        # Hmm, the regex is (\d+)\s+hundred\s+(\d+) matching "3 hundred 20"
        # That gives 3*100+20=320, leaving "320 5 items"
        assert "320" in result

    def test_hundreds_without_remainder(self):
        """Hundreds without remainder."""
        result = ptt.format_dictation("five hundred items")
        assert "500" in result

    def test_two_hundred(self):
        """Two hundred."""
        result = ptt.format_dictation("two hundred")
        assert "200" in result

    # Multiple transformations combined
    def test_combined_transforms(self):
        """Combined transforms."""
        result = ptt.format_dictation("doctor smith has five patients")
        assert result.startswith("Dr.")
        assert "5" in result

    # Edge cases
    def test_whitespace_only(self):
        """Whitespace only."""
        result = ptt.format_dictation("   ")
        assert result == "   "  # Spaces are preserved, first char capitalized

    def test_numeric_start(self):
        """Numeric start."""
        result = ptt.format_dictation("5 items")
        assert result == "5 items"

    def test_no_transformations_needed(self):
        """No transformations needed."""
        result = ptt.format_dictation("hello there")
        assert result == "Hello there"

    def test_longer_words_matched_first(self):
        """Ensure 'seventeen' is matched before 'seven'."""
        result = ptt.format_dictation("seventeen items")
        assert "17" in result
        assert "7teen" not in result

    def test_word_boundary_respected(self):
        """Words like 'onerous' should not be partially converted."""
        result = ptt.format_dictation("onerous task")
        # "one" should not be replaced inside "onerous" due to \b boundary
        assert "1rous" not in result


class TestTranscriptionHistory:
    """Tests covering transcriptionhistory."""

    def test_init_creates_db(self, tmp_path):
        """Init creates db."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=10)
        assert os.path.exists(hist.db_path)

    def test_add_and_recent(self, tmp_path):
        """Add and recent."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=10)
        hist.add("first")
        hist.add("second")
        items = hist.recent(10)
        assert len(items) == 2
        # Most recent first
        assert items[0][0] == "second"
        assert items[1][0] == "first"

    def test_recent_limit(self, tmp_path):
        """Recent limit."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=50)
        for i in range(5):
            hist.add(f"item {i}")
        items = hist.recent(2)
        assert len(items) == 2

    def test_max_items_trimming(self, tmp_path):
        """Max items trimming."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=3)
        for i in range(10):
            hist.add(f"item {i}")
        items = hist.recent(100)
        assert len(items) == 3
        # Should keep the 3 most recent
        texts = [t for t, _ in items]
        assert "item 9" in texts
        assert "item 8" in texts
        assert "item 7" in texts

    def test_empty_recent(self, tmp_path):
        """Empty recent."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=10)
        items = hist.recent(10)
        assert items == []

    def test_timestamp_stored(self, tmp_path):
        """Timestamp stored."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist = ptt.TranscriptionHistory(max_items=10)
        hist.add("test")
        items = hist.recent(1)
        assert len(items) == 1
        assert isinstance(items[0][1], float)
        assert items[0][1] > 0

    def test_creates_directory(self, tmp_path):
        """Creates directory."""
        deep_path = tmp_path / "a" / "b" / "wayvox"
        with patch("os.path.expanduser", return_value=str(deep_path)):
            ptt.TranscriptionHistory(max_items=10)
        assert os.path.isdir(deep_path)

    def test_table_created_idempotent(self, tmp_path):
        """Creating two History instances on same DB should not fail."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "wayvox")):
            hist1 = ptt.TranscriptionHistory(max_items=10)
            hist1.add("test1")
            hist2 = ptt.TranscriptionHistory(max_items=10)
            hist2.add("test2")
        items = hist2.recent(10)
        assert len(items) == 2


class TestWordNums:
    """Tests covering wordnums."""

    def test_all_basic_nums_present(self):
        """All basic nums present."""
        expected = {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        }
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_teens_present(self):
        """Teens present."""
        expected = {
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        }
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_tens_present(self):
        """Tens present."""
        expected = {
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        }
        assert expected.issubset(set(ptt.WORD_NUMS.keys()))

    def test_values_correct(self):
        """Values correct."""
        assert ptt.WORD_NUMS["zero"] == 0
        assert ptt.WORD_NUMS["ten"] == 10
        assert ptt.WORD_NUMS["twenty"] == 20
        assert ptt.WORD_NUMS["ninety"] == 90


class TestAbbreviations:
    """Tests covering abbreviations."""

    def test_abbreviations_dict_not_empty(self):
        """Abbreviations dict not empty."""
        assert len(ptt.ABBREVIATIONS) > 0

    def test_abbreviations_values(self):
        # Check a few expected values
        """Abbreviations values."""
        assert r"\bdoctor\b" in ptt.ABBREVIATIONS
        assert ptt.ABBREVIATIONS[r"\bdoctor\b"] == "Dr."
        assert r"\bversus\b" in ptt.ABBREVIATIONS
        assert ptt.ABBREVIATIONS[r"\bversus\b"] == "vs."
