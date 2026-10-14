"""Диалог художников с экспертом."""

from .records import QuestionAnswer


class Dialogue:
    """Все вопросы художников и ответы эксперта за время работы."""

    def __init__(self, shared_questions: bool):
        self.shared_questions = shared_questions
        self.entries: list[QuestionAnswer] = []

    def add(self, artist_id: str, question: str, answer: str) -> None:
        self.entries.append(QuestionAnswer(artist_id, question, answer))

    def visible_to(self, artist_id: str) -> list[QuestionAnswer]:
        """Что видит конкретный художник — зависит от режима shared_questions."""
        return [entry for entry in self.entries
                if self.shared_questions or entry.artist_id == artist_id]

    @staticmethod
    def to_text(entries: list[QuestionAnswer]) -> str:
        """Превратить записи диалога в читаемый текст для промпта."""
        if not entries:
            return "Пока ничего не согласовано."
        return "\n".join(f"Художник {entry.artist_id}: {entry.question}\nЭксперт: {entry.answer}"
                         for entry in entries)
