import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, fsolve
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Функції для апроксимації кривих попиту і пропозиції
def demand_func(p, a, b):
    return a * np.power(p, -b)


def supply_func(p, c, d):
    return c * np.power(p, d)


# Функція для знаходження точки ринкової рівноваги
def equilibrium(p, a_demand, b_demand, c_supply, d_supply):
    Qd = demand_func(p, a_demand, b_demand)
    Qs = supply_func(p, c_supply, d_supply)
    return Qs - Qd


# Щоб дослідити стабільність рівноваги, ми повинні дивитися на крутизну (похідні) кривих попиту та пропозиції
# в точці рівноваги. Рівновага стабільна, якщо крутизна кривої пропозиції більша за крутизну кривої попиту
# в точці рівноваги (тобто, |dQs/dP| > |dQd/dP|).

# Визначимо функції для похідних
def demand_derivative(p, a, b):
    return -a * b * np.power(p, -(b + 1))


def supply_derivative(p, c, d):
    return c * d * np.power(p, (d - 1))


# Дугова еластичність обчислюється як відсоткове зміщення кількості, ділене на відсоткове зміщення ціни
# для двох точок на кривій. Формула для дугової еластичності:
# E = [(Q2 - Q1) / ((Q2 + Q1) / 2)] / [(P2 - P1) / ((P2 + P1) / 2)]

def arc_elasticity(Q1, Q2, P1, P2):
    quantity_percent_change = (Q2 - Q1) / ((Q2 + Q1) / 2)
    price_percent_change = (P2 - P1) / ((P2 + P1) / 2)
    return quantity_percent_change / price_percent_change


# Для моделювання впливу субсидії необхідно врахувати, що субсидія зменшує витрати виробника, що призводить
# до зміщення кривої пропозиції вліво на величину субсидії.

# Введемо субсидію 0.5 до функції пропозиції, зменшивши ціну виробника
def adjusted_supply_func(p, subsidy, c, d):
    return supply_func(p - subsidy, c, d)


# Дані цін, попиту та пропозиції
prices = [1, 1.25, 1.57, 1.81, 2.09, 2.45, 2.8, 3.19, 3.58, 3.85, 4.5, 5]
demand = [280, 245, 190, 141, 135, 110, 95, 65, 58, 44, 21, 10]
supply = [5, 20, 51, 89, 120, 153, 180, 201, 215, 228, 240, 248]

# Створення DataFrame
market_data = pd.DataFrame({
    'Price': prices,
    'Demand': demand,
    'Supply': supply
})

# Підбір параметрів для кривих попиту та пропозиції
popt_demand, pcov_demand = curve_fit(demand_func, market_data['Price'], market_data['Demand'])
popt_supply, pcov_supply = curve_fit(supply_func, market_data['Price'], market_data['Supply'])

# Виведення параметрів
a_demand, b_demand = popt_demand
c_supply, d_supply = popt_supply

# Построєння кривих попиту та пропозиції
price_range = np.linspace(min(prices), max(prices), 100)
demand_curve = demand_func(price_range, a_demand, b_demand)
supply_curve = supply_func(price_range, c_supply, d_supply)

# Знаходження ціни та кількості в точці ринкової рівноваги
equilibrium_price = fsolve(equilibrium, 2, args=(a_demand, b_demand, c_supply, d_supply))[0]
equilibrium_quantity = supply_func(equilibrium_price, c_supply, d_supply)

# Обчислення похідних у точці рівноваги
demand_slope = demand_derivative(equilibrium_price, a_demand, b_demand)
supply_slope = supply_derivative(equilibrium_price, c_supply, d_supply)

# Виведення результатів
print('Похідна кривої попиту ', demand_slope)
print('Похідна кривої пропозиції ', supply_slope)
print('Нестабільна ', supply_slope > demand_slope)

# Отримаємо кількості та ціни для першого і останнього спостереження для попиту і пропозиції
Q1_demand, Q2_demand = market_data['Demand'].iloc[0], market_data['Demand'].iloc[-1]
Q1_supply, Q2_supply = market_data['Supply'].iloc[0], market_data['Supply'].iloc[-1]
P1, P2 = market_data['Price'].iloc[0], market_data['Price'].iloc[-1]

# Обчислюємо дугову еластичність для попиту і пропозиції
arc_elasticity_demand = arc_elasticity(Q1_demand, Q2_demand, P1, P2)
arc_elasticity_supply = arc_elasticity(Q1_supply, Q2_supply, P1, P2)

print('Дугова еластичність для попиту ', arc_elasticity_demand)
print('Дугова еластичність для пропозиції ', arc_elasticity_supply)

# Знаходження нової рівноважної ціни та кількості з урахуванням субсидії
subsidy = 0.5
new_equilibrium_price = fsolve(lambda p: equilibrium(p, a_demand, b_demand, c_supply, d_supply), equilibrium_price + subsidy)[0]
new_equilibrium_quantity = adjusted_supply_func(new_equilibrium_price, subsidy, c_supply, d_supply)

# Визначимо ціни споживача та виробника
consumer_price = new_equilibrium_price
producer_price = new_equilibrium_price - subsidy


# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(market_data['Price'], market_data['Demand'], label='Demand data', color='blue')
plt.scatter(market_data['Price'], market_data['Supply'], label='Supply data', color='red')
plt.plot(price_range, demand_curve, label=f'Demand curve: $Q_d = {a_demand:.2f}P^{{-{b_demand:.2f}}}$', color='blue')
plt.plot(price_range, supply_curve, label=f'Supply curve: $Q_s = {c_supply:.2f}P^{{{d_supply:.2f}}}$', color='red')
plt.scatter(equilibrium_price, equilibrium_quantity, color='green', label='Equilibrium Point', zorder=5)
plt.title('Market Demand and Supply Curves')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.show()
