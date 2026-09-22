#ifdef _WIN32
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#endif

#include "app_ui.hpp"
#include <string>

#ifdef _WIN32
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(linker, \
    "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' " \
    "version='6.0.0.0' processorArchitecture='*' " \
    "publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif

namespace videoai {
namespace {
#ifdef _WIN32
enum {
    IDC_BTN_CAMERA = 101, IDC_BTN_VIDEO = 102, IDC_BTN_START = 103,
    IDC_STATUS = 104, IDC_RADIO_OFFICIAL = 105, IDC_RADIO_TRT = 106, IDC_MODEL_STATUS = 107,
};
struct LaunchState {
    LaunchRequest request;
    HWND status = nullptr, modelStatus = nullptr, startBtn = nullptr;
    HWND radioOnnx = nullptr, radioTrt = nullptr;
    bool hasSource = false, confirmed = false;
};
void setStatus(HWND hwnd, const wchar_t* text) { SetWindowTextW(hwnd, text); }
void syncModel(LaunchState* state) {
    if (!state) return;
    const bool official = state->radioOnnx &&
        (SendMessageW(state->radioOnnx, BM_GETCHECK, 0, 0) == BST_CHECKED);
    state->request.modelKind = official ? ModelKind::OfficialOnnx : ModelKind::TensorRT;
    if (state->modelStatus) {
        setStatus(state->modelStatus, official
            ? L"\u6a21\u578b\uff1a\u5b98\u7f51\u539f\u6a21\u578b yolo11n.onnx"
            : L"\u6a21\u578b\uff1aTensorRT+CUDA \u4f18\u5316 yolo11n.engine");
    }
}
std::wstring acpToWide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring w(n - 1, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, w.data(), n);
    return w;
}
LRESULT CALLBACK launchProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<LaunchState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_COMMAND: {
            if (!state) break;
            const int id = LOWORD(wParam);
            if (id == IDC_BTN_CAMERA) {
                state->request.action = LaunchAction::Camera;
                state->request.videoPath.clear();
                state->hasSource = true;
                EnableWindow(state->startBtn, TRUE);
                setStatus(state->status, L"\u5df2\u63a5\u5165\uff1a\u6444\u50cf\u5934 0");
            } else if (id == IDC_BTN_VIDEO) {
                std::string path = openVideoFileDialog();
                if (path.empty()) break;
                state->request.action = LaunchAction::Video;
                state->request.videoPath = path;
                state->hasSource = true;
                EnableWindow(state->startBtn, TRUE);
                std::wstring tip = L"\u5df2\u9009\u62e9\u89c6\u9891\uff1a" + acpToWide(path);
                setStatus(state->status, tip.c_str());
            } else if (id == IDC_RADIO_OFFICIAL || id == IDC_RADIO_TRT) {
                syncModel(state);
            } else if (id == IDC_BTN_START && state->hasSource) {
                syncModel(state);
                state->confirmed = true;
                DestroyWindow(hwnd);
            }
            return 0;
        }
        case WM_CLOSE: DestroyWindow(hwnd); return 0;
        case WM_DESTROY: PostQuitMessage(0); return 0;
        default: break;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}
#endif
}  // namespace

std::string openVideoFileDialog() {
#ifdef _WIN32
    char fileBuf[MAX_PATH] = {0};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "Video Files\0*.mp4;*.avi;*.mov;*.mkv;*.flv;*.webm;*.m4v\0All Files\0*.*\0";
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select video";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return std::string(fileBuf);
#endif
    return {};
}

void showErrorDialog(const std::string& message) {
#ifdef _WIN32
    MessageBoxA(nullptr, message.c_str(), "videoai", MB_OK | MB_ICONERROR);
#else
    (void)message;
#endif
}

LaunchRequest showLaunchWindow() {
#ifdef _WIN32
    LaunchState state;
    const wchar_t* cls = L"videoai_launch";
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = launchProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.lpszClassName = cls;
    RegisterClassExW(&wc);
    const int w = 500, h = 390;
    const int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
    const int y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
    HWND hwnd = CreateWindowExW(0, cls, L"videoai  \u76ee\u6807\u68c0\u6d4b",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        x, y, w, h, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) { LaunchRequest req; req.action = LaunchAction::Quit; return req; }
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&state));
    HFONT font = CreateFontW(-17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei UI");
    auto add = [&](const wchar_t* type, const wchar_t* text, int id,
                   int bx, int by, int bw, int bh, DWORD extra) -> HWND {
        HWND c = CreateWindowExW(0, type, text, WS_CHILD | WS_VISIBLE | extra,
            bx, by, bw, bh, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
            wc.hInstance, nullptr);
        SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return c;
    };
    add(L"STATIC", L"1. \u9009\u62e9\u8f93\u5165", 0, 24, 16, 440, 24, 0);
    add(L"BUTTON", L"\u6253\u5f00\u6444\u50cf\u5934", IDC_BTN_CAMERA, 24, 44, 210, 40, BS_PUSHBUTTON);
    add(L"BUTTON", L"\u4e0a\u4f20\u89c6\u9891", IDC_BTN_VIDEO, 250, 44, 210, 40, BS_PUSHBUTTON);
    state.status = add(L"STATIC", L"\u5f53\u524d\uff1a\u672a\u9009\u62e9", IDC_STATUS, 24, 92, 440, 24, 0);
    add(L"STATIC", L"2. \u9009\u62e9\u6a21\u578b", 0, 24, 128, 440, 24, 0);
    state.radioOnnx = add(L"BUTTON", L"\u5b98\u7f51\u539f\u6a21\u578b  yolo11n.onnx",
        IDC_RADIO_OFFICIAL, 24, 156, 436, 28, BS_AUTORADIOBUTTON | WS_GROUP);
    state.radioTrt = add(L"BUTTON", L"TensorRT+CUDA \u4f18\u5316  yolo11n.engine",
        IDC_RADIO_TRT, 24, 188, 436, 28, BS_AUTORADIOBUTTON);
    SendMessageW(state.radioTrt, BM_SETCHECK, BST_CHECKED, 0);
    state.modelStatus = add(L"STATIC", L"", IDC_MODEL_STATUS, 24, 222, 440, 24, 0);
    syncModel(&state);
    state.startBtn = add(L"BUTTON", L"\u5f00\u59cb\u68c0\u6d4b", IDC_BTN_START, 24, 268, 436, 48, BS_DEFPUSHBUTTON);
    EnableWindow(state.startBtn, FALSE);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    DeleteObject(font);
    UnregisterClassW(cls, wc.hInstance);
    if (!state.confirmed) state.request.action = LaunchAction::Quit;
    return state.request;
#else
    LaunchRequest req; req.action = LaunchAction::Quit; return req;
#endif
}
}  // namespace videoai