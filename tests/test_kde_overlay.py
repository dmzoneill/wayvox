"""Widget tests for the overlays in kde-plasmoid/wayvox_kde_service.py.

tests.kde_support loads that module by file path and provides qapp.
"""

import time
from unittest.mock import patch

import pytest

from tests.kde_support import kde_svc

pytest_plugins = ["tests.kde_support"]


@pytest.mark.usefixtures("qapp")
class TestOverlayCard:
    """Checks OverlayCard labels, readiness, and inherited hover handlers."""

    def test_init_ready(self):
        """A ready card shows the tone title and the supplied body."""
        card = kde_svc.OverlayCard("diplomatic", "Hello world", ready=True)
        assert card.tone == "diplomatic"
        assert card.card_text == "Hello world"
        assert card.ready is True
        assert card.text_label.text() == "Hello world"
        assert card.tone_label.text() == "Diplomatic"

    def test_init_not_ready(self):
        """A card that is not ready keeps the processing label."""
        card = kde_svc.OverlayCard("casual", "Processing...", ready=False)
        assert card.ready is False
        assert card.tone_label.text() == "Casual"

    def test_update_text(self):
        """Updating the body marks the card ready and shows the new text."""
        card = kde_svc.OverlayCard("formal", "Loading...", ready=False)
        assert card.ready is False
        card.update_text("This is the rewritten text.")
        assert card.ready is True
        assert card.card_text == "This is the rewritten text."
        assert card.text_label.text() == "This is the rewritten text."

    def test_update_style_called_on_init(self):
        """The card stylesheet includes the button and radius rules."""
        card = kde_svc.OverlayCard("test", "text", ready=False)
        style = card.styleSheet()
        assert "QPushButton" in style
        assert "border-radius" in style

    def test_tone_capitalization(self):
        """The tone label capitalizes the first character of the tone name."""
        card = kde_svc.OverlayCard("summarize", "text")
        assert card.tone_label.text() == "Summarize"

    def test_enter_leave_events(self):
        """The base hover handlers accept an event without a local override."""
        card = kde_svc.OverlayCard("test", "text", ready=True)
        card.enterEvent(None)
        card.leaveEvent(None)


@pytest.mark.usefixtures("qapp")
class TestLanguageBuddyOverlay:
    """Checks the rewrite overlay cards, timer, and selection signals."""

    def test_show_overlay_creates_cards(self):
        """Showing the overlay creates a ready original card and pending tones."""
        overlay = kde_svc.LanguageBuddyOverlay()
        tones = ["diplomatic", "professional"]
        overlay.show_overlay(tones, "Hello world", timeout_sec=30)

        assert "original" in overlay.cards
        assert "diplomatic" in overlay.cards
        assert "professional" in overlay.cards
        assert len(overlay.cards) == 3

        assert overlay.cards["original"].ready is True
        assert overlay.cards["original"].card_text == "Hello world"
        assert overlay.cards["diplomatic"].ready is False
        assert overlay.cards["professional"].ready is False
        overlay.dismiss()

    def test_show_overlay_with_timeout(self):
        """A positive timeout starts the single-shot dismiss timer."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["casual"], "Test", timeout_sec=10)
        assert overlay.timer.isActive()
        overlay.dismiss()

    def test_show_overlay_zero_timeout(self):
        """A zero timeout leaves the dismiss timer stopped."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["casual"], "Test", timeout_sec=0)
        assert not overlay.timer.isActive()
        overlay.dismiss()

    def test_update_card_existing(self):
        """Updating a known tone marks that card ready with the new text."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "Original text")
        overlay.update_card("diplomatic", "Rewritten text")

        card = overlay.cards["diplomatic"]
        assert card.ready is True
        assert card.card_text == "Rewritten text"
        overlay.dismiss()

    def test_update_card_missing_tone(self):
        """Updating a tone that was not shown does not raise."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "text")
        overlay.update_card("nonexistent", "text")
        overlay.dismiss()

    def test_on_card_click_ready(self):
        """Clicking a ready tone card emits that card's text."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")
        overlay.update_card("diplomatic", "rewritten")

        selected = []
        overlay.card_selected.connect(selected.append)
        overlay.on_card_click("diplomatic")

        assert len(selected) == 1
        assert selected[0] == "rewritten"

    def test_on_card_click_not_ready(self):
        """Clicking a card that is still processing does not emit."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")

        selected = []
        overlay.card_selected.connect(selected.append)
        overlay.on_card_click("diplomatic")

        assert len(selected) == 0
        overlay.dismiss()

    def test_on_card_click_missing_tone(self):
        """Clicking a tone that is not on the overlay does not emit."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")

        selected = []
        overlay.card_selected.connect(selected.append)
        overlay.on_card_click("nonexistent")

        assert len(selected) == 0
        overlay.dismiss()

    def test_on_select_emits_signal(self):
        """Selecting text emits that text on the card_selected signal."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "original")

        selected = []
        overlay.card_selected.connect(selected.append)
        overlay.on_select("chosen text")

        assert selected == ["chosen text"]

    def test_dismiss(self):
        """Dismiss stops the timer, hides the window, and drops the cards."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["diplomatic"], "text", timeout_sec=30)
        assert overlay.timer.isActive()

        overlay.dismiss()
        assert not overlay.timer.isActive()
        assert overlay.isHidden()
        assert len(overlay.cards) == 0

    def test_clear_removes_cards(self):
        """Clear removes the original card and every tone card."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["a", "b"], "text")
        assert len(overlay.cards) == 3
        overlay.clear()
        assert len(overlay.cards) == 0

    def test_position_with_screen(self):
        """Positioning against a real screen does not raise."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["test"], "text")
        overlay.position()
        overlay.dismiss()

    def test_position_without_screen(self):
        """Position returns immediately when no primary screen exists."""
        overlay = kde_svc.LanguageBuddyOverlay()
        with patch.object(kde_svc.QGuiApplication, "primaryScreen", return_value=None):
            overlay.position()

    def test_show_overlay_replaces_previous(self):
        """Showing the overlay again replaces the previous set of cards."""
        overlay = kde_svc.LanguageBuddyOverlay()
        overlay.show_overlay(["tone1"], "first")
        assert "tone1" in overlay.cards

        overlay.show_overlay(["tone2"], "second")
        assert "tone2" in overlay.cards
        assert "tone1" not in overlay.cards
        overlay.dismiss()


@pytest.mark.usefixtures("qapp")
class TestHistoryOverlay:
    """Checks history rows, the dismiss timer, and typing the selection."""

    def test_show_history(self):
        """Showing history with a timeout starts the dismiss timer."""
        overlay = kde_svc.HistoryOverlay()
        items = [
            {"text": "Hello world", "ts": time.time() - 30},
            {"text": "Second item", "ts": time.time() - 3600},
        ]
        overlay.show_history(items, timeout_sec=10)
        assert overlay.timer.isActive()
        overlay.dismiss()

    def test_show_history_long_text_truncated(self):
        """A long history row is shown without raising."""
        overlay = kde_svc.HistoryOverlay()
        long_text = "x" * 200
        items = [{"text": long_text, "ts": time.time()}]
        overlay.show_history(items)
        overlay.dismiss()

    def test_show_history_zero_timeout(self):
        """A zero history timeout leaves the dismiss timer stopped."""
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "test", "ts": time.time()}]
        overlay.show_history(items, timeout_sec=0)
        assert not overlay.timer.isActive()
        overlay.dismiss()

    def test_dismiss(self):
        """Dismiss stops the history timer and hides the window."""
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "test", "ts": time.time()}]
        overlay.show_history(items, timeout_sec=10)
        overlay.dismiss()
        assert not overlay.timer.isActive()
        assert overlay.isHidden()

    def test_on_select_types_text(self):
        """Selecting a history row types that row's full text."""
        overlay = kde_svc.HistoryOverlay()
        with patch.object(kde_svc, "type_text") as mock_type:
            overlay.on_select("typed text")
        mock_type.assert_called_once_with("typed text")

    def test_position_with_screen(self):
        """Positioning the history overlay against a screen does not raise."""
        overlay = kde_svc.HistoryOverlay()
        overlay.position()

    def test_position_without_screen(self):
        """History positioning returns immediately when no screen exists."""
        overlay = kde_svc.HistoryOverlay()
        with patch.object(kde_svc.QGuiApplication, "primaryScreen", return_value=None):
            overlay.position()

    def test_clear(self):
        """Clear stops the history timer after rows have been shown."""
        overlay = kde_svc.HistoryOverlay()
        items = [{"text": "a", "ts": time.time()}, {"text": "b", "ts": time.time()}]
        overlay.show_history(items)
        overlay.clear()
        assert not overlay.timer.isActive()
        overlay.dismiss()

    def test_show_replaces_previous(self):
        """Showing history again replaces the previous rows without raising."""
        overlay = kde_svc.HistoryOverlay()
        overlay.show_history([{"text": "first", "ts": time.time()}])
        overlay.show_history([{"text": "second", "ts": time.time()}])
        overlay.dismiss()


@pytest.mark.usefixtures("qapp")
class TestTimeAgoRecent:
    """Checks time_ago for timestamps newer than one hour."""

    def test_just_now_0_seconds(self, history_overlay):
        """A timestamp from this instant is described as just now."""
        assert history_overlay.time_ago(time.time()) == "just now"

    def test_just_now_30_seconds(self, history_overlay):
        """A timestamp from thirty seconds ago is described as just now."""
        assert history_overlay.time_ago(time.time() - 30) == "just now"

    def test_just_now_59_seconds(self, history_overlay):
        """A timestamp from fifty-nine seconds ago is described as just now."""
        assert history_overlay.time_ago(time.time() - 59) == "just now"

    def test_1_minute(self, history_overlay):
        """A timestamp from one minute ago is described as one minute."""
        assert history_overlay.time_ago(time.time() - 60) == "1m ago"

    def test_2_minutes(self, history_overlay):
        """A timestamp from two minutes ago is described as two minutes."""
        assert history_overlay.time_ago(time.time() - 120) == "2m ago"

    def test_30_minutes(self, history_overlay):
        """A timestamp from thirty minutes ago is described as thirty minutes."""
        assert history_overlay.time_ago(time.time() - 1800) == "30m ago"

    def test_59_minutes(self, history_overlay):
        """A timestamp from fifty-nine minutes ago stays in the minute bucket."""
        assert history_overlay.time_ago(time.time() - 3599) == "59m ago"

    def test_90_seconds_is_1m(self, history_overlay):
        """Ninety seconds is still described as one minute."""
        assert history_overlay.time_ago(time.time() - 90) == "1m ago"


@pytest.mark.usefixtures("qapp")
class TestTimeAgoHours:
    """Checks time_ago for hour and day buckets."""

    def test_1_hour(self, history_overlay):
        """A timestamp from one hour ago is described as one hour."""
        assert history_overlay.time_ago(time.time() - 3600) == "1h ago"

    def test_2_hours(self, history_overlay):
        """A timestamp from two hours ago is described as two hours."""
        assert history_overlay.time_ago(time.time() - 7200) == "2h ago"

    def test_23_hours(self, history_overlay):
        """A timestamp from twenty-three hours ago stays in the hour bucket."""
        assert history_overlay.time_ago(time.time() - 82800) == "23h ago"

    def test_5400_seconds_is_1h(self, history_overlay):
        """Ninety minutes is described as one hour because it passed 3600."""
        assert history_overlay.time_ago(time.time() - 5400) == "1h ago"

    def test_1_day(self, history_overlay):
        """A timestamp from one day ago is described as one day."""
        assert history_overlay.time_ago(time.time() - 86400) == "1d ago"

    def test_7_days(self, history_overlay):
        """A timestamp from seven days ago is described as seven days."""
        assert history_overlay.time_ago(time.time() - 604800) == "7d ago"

    def test_30_days(self, history_overlay):
        """A timestamp from thirty days ago is described as thirty days."""
        assert history_overlay.time_ago(time.time() - 2592000) == "30d ago"


@pytest.mark.usefixtures("qapp")
class TestTimeAgoBoundaries:
    """Checks time_ago at the bucket edges and for unusual timestamps."""

    def test_future_timestamp(self, history_overlay):
        """A future timestamp has a negative age and is described as just now."""
        result = history_overlay.time_ago(time.time() + 3600)
        assert result == "just now"

    def test_boundary_60_seconds(self, history_overlay):
        """Exactly sixty seconds is the first minute label."""
        result = history_overlay.time_ago(time.time() - 60)
        assert result == "1m ago"

    def test_boundary_3600_seconds(self, history_overlay):
        """Exactly 3600 seconds is the first hour label."""
        result = history_overlay.time_ago(time.time() - 3600)
        assert result == "1h ago"

    def test_boundary_86400_seconds(self, history_overlay):
        """Exactly 86400 seconds is the first day label."""
        result = history_overlay.time_ago(time.time() - 86400)
        assert result == "1d ago"

    def test_very_old_timestamp(self, history_overlay):
        """A timestamp from 365 days ago is described as 365 days."""
        assert history_overlay.time_ago(time.time() - 365 * 86400) == "365d ago"

    def test_zero_timestamp(self, history_overlay):
        """A zero timestamp is old enough to use the day label."""
        result = history_overlay.time_ago(0)
        assert result.endswith("d ago")
