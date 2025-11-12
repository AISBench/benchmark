import sys

from ais_bench.benchmark.utils.logging.logger import AISLogger

logger = AISLogger()

if sys.platform == 'win32':  # Always return win32 for Windows
    # curses is not supported on Windows
    # If you want to use this function in Windows platform
    # you can try `windows_curses` module by yourself
    curses = None
    logger.warning("Curses is not supported on Windows platform. Menu functionality will be limited.")
else:
    import curses


class Menu:
    """A curses menu that allows the user to select one item from each list.

    Args:
        lists (list[list[str]]): A list of lists of strings, where each list
            represents a list of items to be selected from.
        prompts (list[str], optional): A list of prompts to be displayed above
            each list. Defaults to None, in which case each list will be
            displayed without a prompt.
    """

    def __init__(self, lists, prompts=None):
        self.choices_lists = lists
        self.prompts = prompts or ['Please make a selection:'] * len(lists)
        self.choices = []
        self.current_window = []
        logger.debug(f"Menu initialized with {len(lists)} selection lists")

    def draw_menu(self, stdscr, selected_row_idx, offset, max_rows):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        for idx, row in enumerate(self.current_window[offset:offset +
                                                      max_rows]):
            x = w // 2 - len(row) // 2
            y = min(h - 1,
                    idx + 1)  # Ensure y never goes beyond the window height
            if idx == selected_row_idx - offset:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row)
        stdscr.refresh()

    def run(self):
        logger.debug("Starting menu interactive selection")
        curses.wrapper(self.main_loop)
        logger.debug(f"Menu selection completed with {len(self.choices)} choices: {self.choices}")
        return self.choices

    def main_loop(self, stdscr):
        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        h, _ = stdscr.getmaxyx()
        max_rows = h - 2

        for list_idx, (choices, prompt) in enumerate(zip(self.choices_lists, self.prompts)):
            logger.debug(f"Displaying selection list {list_idx + 1}/{len(self.choices_lists)}: {len(choices)} options")
            self.current_window = [prompt] + choices
            current_row_idx = 1
            offset = 0

            while 1:
                self.draw_menu(stdscr, current_row_idx, offset, max_rows)
                key = stdscr.getch()

                if key == curses.KEY_UP and current_row_idx > 1:
                    current_row_idx -= 1
                    if current_row_idx - offset < 1:
                        offset -= 1

                elif key == curses.KEY_DOWN and current_row_idx < len(choices):
                    current_row_idx += 1
                    if current_row_idx - offset > max_rows - 1:
                        offset += 1

                elif key == curses.KEY_ENTER or key in [10, 13]:
                    selected_choice = choices[current_row_idx - 1]
                    self.choices.append(selected_choice)
                    logger.debug(f"User selected: {selected_choice}")
                    break
