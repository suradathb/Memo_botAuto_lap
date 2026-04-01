*** Settings ***
Documentation     Resource file with reusable keywords and variables.
Library           SeleniumLibrary

*** Variables ***
${SERVER}                    devaion.com7tracking.com
${BROWSER}                   chrome
${VALID USER}                GI10038
${VALID PASSWORD}            GI10038
${LOGIN URL}                 https://${SERVER}

${NAVBAR_LOGIN_BTN}          id=dropdownNavbarLink
${USERNAME_INPUT}            id=login-username
${PASSWORD_INPUT}            id=login-password

# ปุ่ม submit ที่อิงจาก password input โดยตรง
${MODAL_SUBMIT_LOGIN_BTN}    xpath=//input[@id='login-password']/ancestor::div[contains(@class,'space-y-6')]//button[normalize-space()='เข้าสู่ระบบ']

# success condition หลัง login
${STOCK_LINK}                xpath=//a[@href='/stock']
${LOGGED_IN_USER_BUTTON}     xpath=//button[@id='dropdownNavbarLink' and not(contains(normalize-space(.),'เข้าสู่ระบบ'))]

*** Keywords ***
Open Browser To Home Page
    Open Browser    ${LOGIN URL}    ${BROWSER}
    Maximize Browser Window
    Set Selenium Timeout    20s
    Set Selenium Implicit Wait    1s
    Wait Until Page Contains Element    ${NAVBAR_LOGIN_BTN}    20s
    Wait For Page Ready

Wait For Page Ready
    Wait Until Keyword Succeeds    20s    1s
    ...    Execute Javascript    return document.readyState === 'complete'

Wait Until Element Clickable
    [Arguments]    ${locator}    ${timeout}=20s
    Wait Until Element Is Visible    ${locator}    ${timeout}
    Wait Until Element Is Enabled    ${locator}    ${timeout}
    Scroll Element Into View    ${locator}
    Sleep    0.5s

Open Login Form
    Wait For Page Ready
    Wait Until Element Clickable    ${NAVBAR_LOGIN_BTN}    20s

    ${clicked}=    Run Keyword And Return Status
    ...    Click Element    ${NAVBAR_LOGIN_BTN}

    IF    not ${clicked}
        Execute Javascript    document.getElementById('dropdownNavbarLink').click();
    END

    Wait Until Element Is Visible    ${USERNAME_INPUT}    20s
    Wait Until Element Is Visible    ${PASSWORD_INPUT}    20s
    Wait Until Element Is Visible    ${MODAL_SUBMIT_LOGIN_BTN}    20s

Input Username
    [Arguments]    ${username}
    Wait Until Element Is Visible    ${USERNAME_INPUT}    20s
    Clear Element Text    ${USERNAME_INPUT}
    Input Text    ${USERNAME_INPUT}    ${username}
    Execute Javascript
    ...    const el = document.getElementById('login-username');
    ...    el.focus();
    ...    el.dispatchEvent(new Event('input', { bubbles: true }));
    ...    el.dispatchEvent(new Event('change', { bubbles: true }));

Input Password
    [Arguments]    ${password}
    Wait Until Element Is Visible    ${PASSWORD_INPUT}    20s
    Clear Element Text    ${PASSWORD_INPUT}
    Input Text    ${PASSWORD_INPUT}    ${password}
    Execute Javascript
    ...    const el = document.getElementById('login-password');
    ...    el.focus();
    ...    el.dispatchEvent(new Event('input', { bubbles: true }));
    ...    el.dispatchEvent(new Event('change', { bubbles: true }));
    ...    el.dispatchEvent(new Event('blur', { bubbles: true }));

Enable Submit If Needed
    ${disabled}=    Execute Javascript
    ...    const btn = document.evaluate("//input[@id='login-password']/ancestor::div[contains(@class,'space-y-6')]//button[normalize-space()='เข้าสู่ระบบ']", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    ...    return btn ? btn.disabled : null;
    Log To Console    Modal submit disabled before click: ${disabled}

    IF    '${disabled}' == 'True'
        Press Keys    ${PASSWORD_INPUT}    TAB
        Sleep    1s
        ${disabled_after_tab}=    Execute Javascript
        ...    const btn = document.evaluate("//input[@id='login-password']/ancestor::div[contains(@class,'space-y-6')]//button[normalize-space()='เข้าสู่ระบบ']", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        ...    return btn ? btn.disabled : null;
        Log To Console    Modal submit disabled after TAB: ${disabled_after_tab}
    END

Submit Login
    # สำคัญ: ห้ามปิด overlay ตรงนี้ เพราะ overlay นี้คือ login modal เอง
    Enable Submit If Needed
    Wait Until Element Clickable    ${MODAL_SUBMIT_LOGIN_BTN}    20s

    ${clicked}=    Run Keyword And Return Status
    ...    Click Element    ${MODAL_SUBMIT_LOGIN_BTN}

    IF    not ${clicked}
        Log To Console    Normal modal submit click failed. Using JavaScript click.
        Execute Javascript
        ...    const btn = document.evaluate("//input[@id='login-password']/ancestor::div[contains(@class,'space-y-6')]//button[normalize-space()='เข้าสู่ระบบ']", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        ...    if (btn) btn.click();
    END

    Sleep    3s

Wait For Login Success On Same Page
    Wait For Page Ready
    Wait Until Keyword Succeeds    30s    2s    Check Login UI

Check Login UI
    ${stock}=    Run Keyword And Return Status
    ...    Page Should Contain Element    ${STOCK_LINK}

    ${user}=    Run Keyword And Return Status
    ...    Page Should Contain Element    ${LOGGED_IN_USER_BUTTON}

    Log To Console    stock=${stock} | user=${user}

    IF    ${stock}
        RETURN
    END

    IF    ${user}
        RETURN
    END

    Fail    UI not updated yet

Page Should Stay On Home After Login
    ${current_url}=    Get Location
    ${normalized_url}=    Evaluate    """${current_url}""".strip().rstrip('/')
    ${normalized_home}=   Evaluate    """${LOGIN URL}""".strip().rstrip('/')
    Should Be Equal    ${normalized_url}    ${normalized_home}