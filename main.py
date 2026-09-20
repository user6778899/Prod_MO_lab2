"""Вторая лабораторная работа: стохастическая оптимизация."""
from experiments import create_output_dirs, run_all, study_epsilon, compare_with_previous
from visualization import create_all_plots


def main():
    create_output_dirs()

    print("1. Основные эксперименты...")
    print(run_all(eps=1e-8).to_string(index=False))

    print("\n2. Исследование epsilon...")
    print(study_epsilon().to_string(index=False))

    print("\n3. Полное сравнение с методами предыдущей работы...")
    print(compare_with_previous(eps=1e-8).to_string(index=False))

    print("\n4. Построение графиков...")
    for path in create_all_plots(eps=1e-8):
        print(path)

    print("\nГотово. Результаты: results/ ; графики: plots/")


if __name__ == "__main__":
    main()
